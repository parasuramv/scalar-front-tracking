"""Event queue shared by the real-line and circle drivers (internal).

A front moves on a straight line between collisions, so it never needs to be
stepped. During ``run`` each front keeps an anchor: its position ``x`` at the
anchor time ``t0``; the position at time t is ``x + speed * (t - t0)``. Only
the fronts created at an event get new anchors, so an event costs time
proportional to its cluster, not to the number of fronts.

Predicted collisions sit in a binary heap. Front i owns the prediction for
the gap to its right neighbour. Whenever that neighbour changes, front i's
``version`` is bumped and a new prediction pushed; the old heap entry still
exists but no longer matches the version, so it is discarded when it reaches
the top (lazy invalidation). A retired front is bumped the same way.

When ``run`` returns, or raises, every front is moved to its anchor at the
current time again, so ``Front.x`` is correct whenever a user reads it.
"""

from bisect import bisect_left
from itertools import count
import heapq
import math


class AnchoredEventDriver:
    """Shared ``run`` loop. Subclasses supply the geometry of one domain.

    Required attributes: ``fronts`` (ordered list), ``time``, ``event_count``,
    ``max_events``, ``record_history``, ``segments``. Required methods:
    ``_pair(i)`` -> (left, right, gap) or None, ``_resolve_cluster(i)``,
    ``_record_path(x0, t0, speed, t1)`` and ``_normalize()``.
    """

    _event_limit_message = "Event limit reached; inspect data or raise max_events"

    # ------------------------------------------------------------ anchors
    def _position(self, front):
        """Position of an anchored front at the current clock, in closed form."""
        return front.x + front.speed * (self.time - front.t0)

    def _retire(self, front):
        """Invalidate a front's queued prediction and record its final path."""
        front.version += 1
        if self.record_history and self.time > front.t0:
            self._record_path(front.x, front.t0, front.speed, self.time)

    def _settle(self):
        """Move every anchor to the current time; afterwards ``x`` is exact."""
        t, record = self.time, self.record_history
        for front in self.fronts:
            if front.t0 != t:
                if record and t > front.t0:
                    self._record_path(front.x, front.t0, front.speed, t)
                front.x += front.speed * (t - front.t0)
                front.t0 = t
        self._normalize()

    def _normalize(self):
        """Hook for a common coordinate shift after settling (circle only)."""

    # -------------------------------------------------------------- queue
    def _start_queue(self):
        """Anchor all fronts now and predict every adjacent pair, O(F log F).

        Rebuilt at the start of every ``run`` so edits made to ``fronts``
        between runs are respected, exactly as the old full rescan did.
        """
        t = self.time
        for front in self.fronts:
            front.t0 = t
        self._heap, self._seq = [], count()
        self._refresh(range(len(self.fronts)))

    def _refresh(self, indices):
        """Replace the predictions owned by the fronts at these indices."""
        heap, seq, t = self._heap, self._seq, self.time
        for i in indices:
            pair = self._pair(i)
            if pair is None:
                continue
            left, right, gap = pair
            left.version += 1
            closing = left.speed - right.speed
            if closing > 0:
                # Equal or separating speeds never collide, even at the same
                # point. Clamp tiny negative gaps (roundoff) to "now".
                wait = max(0.0, gap / closing)
                # Ties in time go to the leftmost meeting point, then FIFO.
                heapq.heappush(heap, (t + wait, self._position(left) + left.speed * wait,
                                      next(seq), left, left.version))

    def _next_event(self):
        """Earliest valid prediction as (time, owner front), or (inf, None)."""
        heap = self._heap
        if len(heap) > 4 * len(self.fronts) + 64:
            # Too many stale entries: rebuild from the live pairs.
            heap.clear()
            self._refresh(range(len(self.fronts)))
        while heap:
            t, _, _, front, version = heap[0]
            if front.version == version:
                return t, front
            heapq.heappop(heap)
        return math.inf, None

    def _locate(self, front):
        """Index of a live front: binary search on positions, then a local walk.

        Positions at the current time are sorted up to roundoff. Fronts that
        roundoff can misorder nearly coincide, so the walk outward from the
        bisection point meets the target after a few steps.
        """
        fronts, t = self.fronts, self.time
        i = bisect_left(fronts, self._position(front),
                        key=lambda f: f.x + f.speed * (t - f.t0))
        lo, hi, n = i - 1, i, len(fronts)
        while lo >= 0 or hi < n:
            if hi < n:
                if fronts[hi] is front:
                    return hi
                hi += 1
            if lo >= 0:
                if fronts[lo] is front:
                    return lo
                lo -= 1
        raise RuntimeError("Queued front is missing from the front list")

    # ---------------------------------------------------------------- run
    def run(self, final_time):
        """Advance to absolute final_time, resolving events exactly at that time.

        Returns self for chaining. Calling again continues the current solution;
        going backward is rejected. An event-limit error leaves the state at
        its last completed event so it can be inspected or resumed.
        """
        if not math.isfinite(final_time) or final_time < self.time:
            raise ValueError("final_time must be finite and at least the current time")
        self._start_queue()
        try:
            while True:
                t, front = self._next_event()
                if front is None or t > final_time:
                    # No interaction before the observation time: translation.
                    self.time = float(final_time)
                    return self
                if self.event_count >= self.max_events:
                    raise RuntimeError(self._event_limit_message)
                heapq.heappop(self._heap)
                self.time = max(self.time, t)
                self._resolve_cluster(self._locate(front))
                # A separate collision at the same time is the next heap entry.
        finally:
            self._settle()
            self._heap = []  # Nothing queued survives outside run().
