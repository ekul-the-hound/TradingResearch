# ==============================================================================
# test_untestable_ideas_menu.py -- Tests for the Phase 5 untestable-ideas menu
# ==============================================================================
# Convention: import failures are HARD errors, not skips.
# ==============================================================================

import os
import tempfile
import unittest

import algorithm_ideas as ai
from untestable_ideas_menu import (
    UntestableIdeasMenu, MenuItem, MenuGroup, _norm_data_type,
)


class MenuTestBase(unittest.TestCase):
    def setUp(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.bl = ai.IdeaBacklog(db_path=self.db)
        self.menu = UntestableIdeasMenu(backlog=self.bl)

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            p = self.db + suffix
            if os.path.exists(p):
                os.remove(p)

    def _add(self, title, data_needed, **kw):
        return self.bl.capture(title=title, description=kw.get("description", "d"),
                               why_untestable=kw.get("why", "w"),
                               data_needed=data_needed,
                               confidence=kw.get("confidence", ""))


class TestNormalization(unittest.TestCase):
    def test_order_book_variants_merge(self):
        for phrasing in ("order book", "order-book depth", "L2 orderbook",
                         "depth of market", "DOM"):
            self.assertEqual(_norm_data_type(phrasing), "order book")

    def test_funding_variants(self):
        self.assertEqual(_norm_data_type("perp funding rate"), "funding rates")

    def test_options_variants(self):
        self.assertEqual(_norm_data_type("implied vol surface"), "options data")

    def test_empty_is_unspecified(self):
        self.assertEqual(_norm_data_type(""), "unspecified")

    def test_unknown_kept_trimmed(self):
        # An unfamiliar data type is kept (trimmed), not merged into a catch-all.
        out = _norm_data_type("proprietary broker flow ratio")
        self.assertIn("proprietary", out)


class TestMissingDataFilter(MenuTestBase):
    def test_only_missing_data_ideas_appear(self):
        self._add("Needs OB", "order book")
        self._add("No data reason", "")  # untestable for another reason
        items = self.menu.list_items()
        titles = [i.title for i in items]
        self.assertIn("Needs OB", titles)
        self.assertNotIn("No data reason", titles)

    def test_empty_menu_when_no_missing_data(self):
        self._add("Vague", "")
        self.assertEqual(self.menu.list_items(), [])

    def test_promoted_and_discarded_excluded(self):
        iid = self._add("Will discard", "funding rate")
        self.menu.discard(iid)
        titles = [i.title for i in self.menu.list_items()]
        self.assertNotIn("Will discard", titles)


class TestGrouping(MenuTestBase):
    def test_grouped_by_data_type(self):
        self._add("A", "order book")
        self._add("B", "depth of market")   # same bucket as A
        self._add("C", "funding rate")
        groups = self.menu.grouped()
        by_type = {g.data_type: g.count for g in groups}
        self.assertEqual(by_type.get("order book"), 2)
        self.assertEqual(by_type.get("funding rates"), 1)

    def test_largest_group_first(self):
        self._add("A", "order book")
        self._add("B", "order book")
        self._add("C", "funding rate")
        groups = self.menu.grouped()
        self.assertEqual(groups[0].data_type, "order book")  # 2 > 1

    def test_data_types_list(self):
        self._add("A", "order book")
        self._add("C", "options greeks")
        self.assertEqual(set(self.menu.data_types()),
                         {"order book", "options data"})

    def test_filter_by_data_type(self):
        self._add("A", "order book")
        self._add("C", "funding rate")
        ob = self.menu.list_items(data_type="order book")
        self.assertEqual(len(ob), 1)
        self.assertEqual(ob[0].title, "A")


class TestActions(MenuTestBase):
    def test_mark_promising(self):
        iid = self._add("X", "order book")
        self.assertTrue(self.menu.mark_promising(iid))
        # still on the menu (promising is an active status)
        self.assertIn("X", [i.title for i in self.menu.list_items()])

    def test_discard_removes(self):
        iid = self._add("Y", "order book")
        self.assertTrue(self.menu.discard(iid))
        self.assertNotIn("Y", [i.title for i in self.menu.list_items()])

    def test_action_on_missing_idea(self):
        self.assertFalse(self.menu.discard("nonexistent-id"))


class TestRendering(MenuTestBase):
    def test_render_empty(self):
        out = self.menu.render()
        self.assertIn("nothing here", out)

    def test_render_with_items(self):
        self._add("OB Scalper", "order book", confidence="promising")
        out = self.menu.render()
        self.assertIn("ORDER BOOK", out)
        self.assertIn("OB Scalper", out)
        self.assertIn("CANNOT be backtested", out)

    def test_one_line_format(self):
        item = MenuItem(
            idea_id="1", title="Test", description="d",
            data_needed="order book", data_type="order book",
            why_untestable="w", category="uncategorized",
            confidence="promising", status="open")
        line = item.one_line()
        self.assertIn("Test", line)
        self.assertIn("promising", line)
        self.assertIn("order book", line)


class TestToDict(MenuTestBase):
    def test_structured_payload(self):
        self._add("A", "order book")
        self._add("B", "funding rate")
        payload = self.menu.to_dict()
        self.assertEqual(payload["total"], 2)
        self.assertTrue(len(payload["groups"]) >= 2)
        # each group has data_type, count, items
        g = payload["groups"][0]
        self.assertIn("data_type", g)
        self.assertIn("count", g)
        self.assertIn("items", g)


class TestNoBacklog(unittest.TestCase):
    def test_graceful_without_backlog(self):
        # A menu with no backlog must not raise; it renders an unavailable note.
        menu = UntestableIdeasMenu(backlog=None)
        # Force the no-backlog path regardless of whether algorithm_ideas imports.
        menu.backlog = None
        self.assertEqual(menu.list_items(), [])
        self.assertIn("unavailable", menu.render())


if __name__ == "__main__":
    unittest.main(verbosity=2)
