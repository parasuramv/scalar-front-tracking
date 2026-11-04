"""Event queue shared by the real-line and circle drivers (internal).

A front moves on a straight line between collisions, so it never needs to be
stepped. During ``run`` each front keeps an anchor: its position ``x`` at the
anchor time ``t0``; the position at time t is ``x + speed * (t - t0)``. Only
the fronts created at an event get new anchors, so surviving fronts need no
position update during a collision.

During ``run`` the live fronts also form a doubly linked list (``prev`` and
``next``), entered at ``self._head``. A collision replaces its cluster by
relinking, in time proportional to the cluster wherever it sits: no other
front moves in memory, as it would when splicing a Python list.

Predicted collisions sit in a binary heap. A front owns the prediction for
the gap to its right neighbour, and the heap entry holds the owner itself, so
an event needs no search to find it. Whenever that neighbour changes, the
owner's ``version`` is bumped and a new prediction pushed; the old heap entry
still exists but no longer matches the version, so it is discarded when it
reaches the top (lazy invalidation). A retired front is bumped the same way.

When ``run`` returns, or raises, the stored front list is rebuilt from the
links (in place, so references to it stay valid), the links are cleared, and
every front is moved to its anchor at the current time again: ``Front.x`` is
correct whenever a user reads it. Between runs that list is authoritative;
edits made to it are picked up when the next ``run`` relinks. Code that reads
``fronts`` during a run (a subclass hook, say) gets a fresh list built from the
links, at O(F) cost per read; positions in it are anchors, not current.
"""

from itertools import count, pairwise
import heapq
import math


class AnchoredEventDriver:
    """Shared ``run`` loop. Subclasses supply the geometry of one domain.

    Required attributes: ``collection`` (its ``fronts`` list is the ordered
    front list between runs), ``time``, ``event_count``, ``max_events``,
    ``record_history``, ``segments``. Required methods:
    ``_pair(front)`` -> (front, right, gap) or None, ``_resolve_cluster(front)``,
    ``_record_path(x0, t0, speed, t1)`` and ``_normalize()``. ``_cyclic`` says
    whether the last front's right neighbour is the first one.
    """

    _event_limit_message = "Event limit reached; inspect data or raise max_events"
    _cyclic = False
    _running = False  # True while the links, not the list, are authoritative.

    @property
    def fronts(self):
        """Live fronts in spatial order. Use snapshot() for an editable copy.

        Between runs this is the stored list itself. During ``run`` (for
        example from a subclass hook) it is a fresh list built from the links.
        """
        if self._running:
            return list(self._live())
        return self.collection.fronts

    # ------------------------------------------------------------ anchors
    def _position(self, front):
        """Position of an anchored front at the current clock, in closed form."""
        return front.x + front.speed * (self.time - front.t0)

    def _retire(self, front):
        """Invalidate a front's queued prediction, record its final path, unlink it."""
        front.version += 1
        if self.record_history and self.time > front.t0:
            self._record_path(front.x, front.t0, front.speed, self.time)
        front.prev = front.next = None

    def _settle(self):
        """Rebuild ``fronts`` from the links and move every anchor to now."""
        live = list(self._live())
        self.collection.fronts[:] = live
        for front in live:
            front.prev = front.next = None
        self._head, self._count, self._running = None, 0, False
        t, record = self.time, self.record_history
        for front in live:
            if front.t0 != t:
                if record and t > front.t0:
                    self._record_path(front.x, front.t0, front.speed, t)
                front.x += front.speed * (t - front.t0)
                front.t0 = t
        self._normalize()

    def _normalize(self):
        """Hook for a common coordinate shift after settling (circle only)."""

    # -------------------------------------------------------------- links
    def _link(self):
        """Link ``fronts`` in list order; afterwards the links are authoritative."""
        fronts = self.collection.fronts
        for left, right in pairwise(fronts):
            left.next, right.prev = right, left
        if fronts:
            if self._cyclic:
                fronts[-1].next, fronts[0].prev = fronts[0], fronts[-1]
            else:
                fronts[0].prev = fronts[-1].next = None
        self._head = fronts[0] if fronts else None
        self._count = len(fronts)
        self._running = True

    def _live(self):
        """Live fronts in order, starting at the head."""
        front = self._head
        for _ in range(self._count):
            yield front
            front = front.next

    @staticmethod
    def _chain(fronts):
        """Link consecutive fronts; None entries mark open ends."""
        for left, right in pairwise(fronts):
            if left is not None:
                left.next = right
            if right is not None:
                right.prev = left

    @staticmethod
    def _span(first, last):
        """The linked fronts from first to last, inclusive."""
        fronts = [first]
        while fronts[-1] is not last:
            fronts.append(fronts[-1].next)
        return fronts

    # -------------------------------------------------------------- queue
    def _start_queue(self):
        """Anchor and link all fronts now and predict every adjacent pair.

        Rebuilt at the start of every ``run`` so edits made to ``fronts``
        between runs are respected, exactly as the old full rescan did.
        """
        t = self.time
        fronts = self.collection.fronts
        for front in fronts:
            front.t0 = t
        self._heap, self._seq = [], count()
        self._link()
        self._refresh(fronts)

    def _refresh(self, fronts):
        """Replace the predictions owned by these fronts (None entries skipped)."""
        heap, seq, t = self._heap, self._seq, self.time
        for front in fronts:
            if front is None:
                continue
            front.version += 1
            pair = self._pair(front)
            if pair is None:
                continue
            left, right, gap = pair
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
        if len(heap) > 4 * self._count + 64:
            # Too many stale entries: rebuild from the live pairs.
            heap.clear()
            self._refresh(self._live())
        while heap:
            t, _, _, front, version = heap[0]
            if front.version == version:
                return t, front
            heapq.heappop(heap)
        return math.inf, None

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
                self._resolve_cluster(front)
                # A separate collision at the same time is the next heap entry.
        finally:
            self._settle()
            self._heap = []  # Nothing queued survives outside run().
