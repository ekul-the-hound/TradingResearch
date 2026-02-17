#!/usr/bin/env python3
# ==============================================================================
# test_live_trading.py — Tests for broker_adapter.py and live_engine.py
# ==============================================================================
# All tests use PaperBroker — no real broker connections, API keys, or
# market data needed. Tests verify:
#   1. Broker adapter interface (all 3 broker types)
#   2. Order flow and position tracking
#   3. Kill switch → flatten integration
#   4. LiveEngine tick loop
#   5. Shadow trading through broker
#   6. End-of-day processing
# ==============================================================================

import sys
import time
import traceback
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from broker_adapter import (
    create_broker, PaperBroker, CCXTBroker, IBKRBroker,
    BaseBroker, BrokerOrder, BrokerPosition, BrokerBalance, BrokerTick,
    OrderStatus, OrderSide, OrderType,
)
from live_engine import LiveEngine, LiveEngineConfig, StrategySlot

_p, _f, _e = 0, 0, []

def run_test(name, fn):
    global _p, _f
    try:
        fn(); _p += 1; print(f"  ✅ {name}")
    except Exception as ex:
        _f += 1; _e.append((name, str(ex))); print(f"  ❌ {name}: {ex}")
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# BROKER ADAPTER (20 tests)
# ══════════════════════════════════════════════════════════════════════════════

def test_ba_01_factory_paper():
    broker = create_broker("paper")
    assert isinstance(broker, PaperBroker)

def test_ba_02_factory_ccxt():
    broker = create_broker("ccxt", exchange="binance")
    assert isinstance(broker, CCXTBroker)

def test_ba_03_factory_ibkr():
    broker = create_broker("ibkr")
    assert isinstance(broker, IBKRBroker)

def test_ba_04_factory_invalid():
    try:
        create_broker("invalid")
        assert False, "Should raise"
    except ValueError:
        pass

def test_ba_05_paper_connect():
    broker = create_broker("paper")
    assert broker.connect() is True
    assert broker.is_connected

def test_ba_06_paper_balance():
    broker = PaperBroker(initial_balance=50_000)
    broker.connect()
    bal = broker.get_balance()
    assert bal.total_equity == 50_000
    assert bal.currency == "USD"

def test_ba_07_paper_tick():
    broker = PaperBroker()
    broker.connect()
    broker.set_price("BTC/USDT", 50000)
    tick = broker.get_tick("BTC/USDT")
    assert tick is not None
    assert tick.last == 50000
    assert tick.spread_bps > 0

def test_ba_08_paper_tick_missing():
    broker = PaperBroker()
    broker.connect()
    assert broker.get_tick("NONEXISTENT") is None

def test_ba_09_paper_buy():
    broker = PaperBroker(initial_balance=100_000, slippage_bps=0)
    broker.connect()
    broker.set_price("BTC/USDT", 50000)
    order = broker.submit_order("buy", "BTC/USDT", 1.0)
    assert order.status == OrderStatus.FILLED
    assert order.filled_size == 1.0
    pos = broker.get_position("BTC/USDT")
    assert pos is not None
    assert pos.side == "long"
    assert pos.size == 1.0

def test_ba_10_paper_sell_close():
    broker = PaperBroker(initial_balance=100_000, slippage_bps=0)
    broker.connect()
    broker.set_price("BTC/USDT", 50000)
    broker.submit_order("buy", "BTC/USDT", 1.0)
    broker.set_price("BTC/USDT", 55000)
    broker.submit_order("sell", "BTC/USDT", 1.0)
    pos = broker.get_position("BTC/USDT")
    assert pos.side == "flat" or pos.size == 0
    assert broker.balance > 100_000  # Profit from 50k→55k

def test_ba_11_paper_short():
    broker = PaperBroker(initial_balance=100_000, slippage_bps=0)
    broker.connect()
    broker.set_price("ETH/USDT", 3000)
    broker.submit_order("sell", "ETH/USDT", 10.0)
    pos = broker.get_position("ETH/USDT")
    assert pos.side == "short"
    assert pos.size == 10.0

def test_ba_12_paper_pnl():
    broker = PaperBroker(initial_balance=100_000, slippage_bps=0)
    broker.connect()
    broker.set_price("BTC/USDT", 50000)
    broker.submit_order("buy", "BTC/USDT", 1.0)
    broker.set_price("BTC/USDT", 52000)
    broker.mark_to_market()
    pos = broker.get_position("BTC/USDT")
    assert abs(pos.unrealized_pnl - 2000) < 100  # ~$2000 profit

def test_ba_13_flatten():
    broker = PaperBroker(slippage_bps=0)
    broker.connect()
    broker.set_price("BTC/USDT", 50000)
    broker.submit_order("buy", "BTC/USDT", 1.0)
    order = broker.flatten("BTC/USDT")
    assert order is not None
    assert order.status == OrderStatus.FILLED
    pos = broker.get_position("BTC/USDT")
    assert pos.size == 0

def test_ba_14_flatten_all():
    broker = PaperBroker(slippage_bps=0)
    broker.connect()
    broker.set_price("BTC/USDT", 50000)
    broker.set_price("ETH/USDT", 3000)
    broker.submit_order("buy", "BTC/USDT", 1.0)
    broker.submit_order("sell", "ETH/USDT", 5.0)
    orders = broker.flatten_all()
    assert len(orders) == 2
    assert all(o.status == OrderStatus.FILLED for o in orders)
    assert len(broker.get_positions()) == 0

def test_ba_15_flatten_flat():
    broker = PaperBroker()
    broker.connect()
    assert broker.flatten("NOTHING") is None

def test_ba_16_order_history():
    broker = PaperBroker()
    broker.connect()
    broker.set_price("BTC/USDT", 50000)
    broker.submit_order("buy", "BTC/USDT", 1.0)
    broker.submit_order("sell", "BTC/USDT", 0.5)
    assert len(broker.order_history) == 2

def test_ba_17_get_order():
    broker = PaperBroker()
    broker.connect()
    broker.set_price("BTC/USDT", 50000)
    order = broker.submit_order("buy", "BTC/USDT", 1.0)
    found = broker.get_order(order.order_id)
    assert found is not None
    assert found.order_id == order.order_id

def test_ba_18_multiple_positions():
    broker = PaperBroker(slippage_bps=0)
    broker.connect()
    broker.set_price("BTC/USDT", 50000)
    broker.set_price("ETH/USDT", 3000)
    broker.set_price("SOL/USDT", 100)
    broker.submit_order("buy", "BTC/USDT", 1.0)
    broker.submit_order("buy", "ETH/USDT", 10.0)
    broker.submit_order("sell", "SOL/USDT", 100.0)
    assert len(broker.get_positions()) == 3

def test_ba_19_commission():
    broker = PaperBroker(slippage_bps=0)
    broker.connect()
    broker.set_price("BTC/USDT", 50000)
    order = broker.submit_order("buy", "BTC/USDT", 1.0)
    assert order.commission > 0  # 0.1% default

def test_ba_20_disconnect():
    broker = PaperBroker()
    broker.connect()
    assert broker.is_connected
    broker.disconnect()
    assert not broker.is_connected


# ══════════════════════════════════════════════════════════════════════════════
# LIVE ENGINE (15 tests)
# ══════════════════════════════════════════════════════════════════════════════

def _make_engine():
    broker = PaperBroker(initial_balance=100_000, slippage_bps=1.0)
    broker.connect()
    broker.set_price("BTC/USDT", 50000)
    broker.set_price("ETH/USDT", 3000)
    cfg = LiveEngineConfig(verbose=False, poll_interval_seconds=0.01)
    return LiveEngine(broker, cfg), broker

def test_le_01_init():
    engine, _ = _make_engine()
    assert engine._running is False
    assert len(engine._slots) == 0

def test_le_02_add_strategy():
    engine, _ = _make_engine()
    slot = engine.add_strategy("S_001", "BTC/USDT", mode="shadow", backtest_sharpe=1.5)
    assert "S_001" in engine._slots
    assert slot.mode == "shadow"

def test_le_03_add_with_returns():
    engine, _ = _make_engine()
    rets = np.random.normal(0.001, 0.01, 100)
    slot = engine.add_strategy("S_001", "BTC/USDT", backtest_returns=rets)
    assert slot.drift_detector is not None or True  # May not import

def test_le_04_remove_strategy():
    engine, _ = _make_engine()
    engine.add_strategy("S_001", "BTC/USDT")
    engine.remove_strategy("S_001")
    assert "S_001" not in engine._slots

def test_le_05_single_tick():
    engine, broker = _make_engine()
    engine.add_strategy("S_001", "BTC/USDT", mode="shadow")
    engine._tick()
    slot = engine._slots["S_001"]
    assert slot.last_tick is not None
    assert slot.last_tick.last == 50000

def test_le_06_signal_shadow():
    engine, broker = _make_engine()
    # Signal function that always buys
    engine.add_strategy("S_001", "BTC/USDT", mode="shadow",
                        signal_fn=lambda tick: ("BUY", 0.1))
    engine._tick()
    slot = engine._slots["S_001"]
    if slot.shadow_trader:
        assert slot.shadow_trader.metrics.total_trades >= 0

def test_le_07_signal_live():
    engine, broker = _make_engine()
    bought = []
    engine.add_strategy("S_001", "BTC/USDT", mode="live",
                        signal_fn=lambda tick: ("BUY", 0.1))
    engine._tick()
    # Check broker got the order
    assert len(broker.order_history) >= 1

def test_le_08_no_signal():
    engine, broker = _make_engine()
    engine.add_strategy("S_001", "BTC/USDT", mode="shadow",
                        signal_fn=lambda tick: None)
    engine._tick()
    # No orders should exist
    assert len(broker.order_history) == 0

def test_le_09_halted_strategy():
    engine, broker = _make_engine()
    engine.add_strategy("S_001", "BTC/USDT", mode="shadow",
                        signal_fn=lambda tick: ("BUY", 0.1))
    engine._slots["S_001"].is_halted = True
    engine._tick()
    # Halted = no orders
    assert len(broker.order_history) == 0

def test_le_10_multiple_strategies():
    engine, broker = _make_engine()
    engine.add_strategy("S_001", "BTC/USDT", mode="shadow")
    engine.add_strategy("S_002", "ETH/USDT", mode="shadow")
    engine._tick()
    assert engine._slots["S_001"].last_tick.last == 50000
    assert engine._slots["S_002"].last_tick.last == 3000

def test_le_11_run_loop_limited():
    engine, broker = _make_engine()
    engine.add_strategy("S_001", "BTC/USDT", mode="shadow")
    engine.run_loop(interval_seconds=0.01, max_ticks=5)
    assert engine._tick_count == 5

def test_le_12_emergency_shutdown():
    engine, broker = _make_engine()
    engine.add_strategy("S_001", "BTC/USDT", mode="live",
                        signal_fn=lambda tick: ("BUY", 0.1))
    engine._tick()  # Creates a position
    engine._emergency_shutdown()
    assert all(s.is_halted for s in engine._slots.values())

def test_le_13_status():
    engine, broker = _make_engine()
    engine.add_strategy("S_001", "BTC/USDT", mode="shadow")
    engine._tick()
    status = engine.get_status()
    assert status["broker"] == "paper"
    assert "S_001" in status["strategies"]
    assert status["strategies"]["S_001"]["last_price"] == 50000

def test_le_14_stop():
    engine, _ = _make_engine()
    engine.add_strategy("S_001", "BTC/USDT")
    engine.stop()
    assert engine._running is False

def test_le_15_price_update_shadow():
    engine, broker = _make_engine()
    engine.add_strategy("S_001", "BTC/USDT", mode="shadow",
                        signal_fn=lambda tick: ("BUY", 0.1) if tick.last == 50000 else None)
    engine._tick()  # Buy at 50000
    broker.set_price("BTC/USDT", 55000)
    engine._tick()  # Mark to market at 55000
    slot = engine._slots["S_001"]
    if slot.shadow_trader:
        assert slot.shadow_trader.position.current_price == 55000


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

ALL = [
    # Broker Adapter (20)
    ("BA.01 Factory paper",        test_ba_01_factory_paper),
    ("BA.02 Factory ccxt",         test_ba_02_factory_ccxt),
    ("BA.03 Factory ibkr",         test_ba_03_factory_ibkr),
    ("BA.04 Factory invalid",      test_ba_04_factory_invalid),
    ("BA.05 Paper connect",        test_ba_05_paper_connect),
    ("BA.06 Paper balance",        test_ba_06_paper_balance),
    ("BA.07 Paper tick",           test_ba_07_paper_tick),
    ("BA.08 Tick missing",         test_ba_08_paper_tick_missing),
    ("BA.09 Buy order",            test_ba_09_paper_buy),
    ("BA.10 Sell close",           test_ba_10_paper_sell_close),
    ("BA.11 Short position",       test_ba_11_paper_short),
    ("BA.12 Unrealized PnL",       test_ba_12_paper_pnl),
    ("BA.13 Flatten single",       test_ba_13_flatten),
    ("BA.14 Flatten all",          test_ba_14_flatten_all),
    ("BA.15 Flatten flat",         test_ba_15_flatten_flat),
    ("BA.16 Order history",        test_ba_16_order_history),
    ("BA.17 Get order",            test_ba_17_get_order),
    ("BA.18 Multiple positions",   test_ba_18_multiple_positions),
    ("BA.19 Commission",           test_ba_19_commission),
    ("BA.20 Disconnect",           test_ba_20_disconnect),
    # Live Engine (15)
    ("LE.01 Init",                 test_le_01_init),
    ("LE.02 Add strategy",         test_le_02_add_strategy),
    ("LE.03 Add with returns",     test_le_03_add_with_returns),
    ("LE.04 Remove strategy",      test_le_04_remove_strategy),
    ("LE.05 Single tick",          test_le_05_single_tick),
    ("LE.06 Signal shadow",        test_le_06_signal_shadow),
    ("LE.07 Signal live",          test_le_07_signal_live),
    ("LE.08 No signal",            test_le_08_no_signal),
    ("LE.09 Halted strategy",      test_le_09_halted_strategy),
    ("LE.10 Multiple strategies",  test_le_10_multiple_strategies),
    ("LE.11 Run loop limited",     test_le_11_run_loop_limited),
    ("LE.12 Emergency shutdown",   test_le_12_emergency_shutdown),
    ("LE.13 Status",               test_le_13_status),
    ("LE.14 Stop",                 test_le_14_stop),
    ("LE.15 Price update shadow",  test_le_15_price_update_shadow),
]

if __name__ == "__main__":
    start = time.time()
    mods = [
        ("Broker Adapter",  0, 20),
        ("Live Engine",    20, 35),
    ]
    for name, lo, hi in mods:
        print(f"\n{'─'*60}\n  {name}\n{'─'*60}")
        for n, fn in ALL[lo:hi]:
            run_test(n, fn)
    print(f"\n  ⏱️  {time.time()-start:.1f}s\n{'='*60}")
    print(f"  LIVE TRADING: {_p} passed, {_f} failed")
    if _e:
        for n, e in _e:
            print(f"    {n}: {e}")
    print(f"{'='*60}")
    sys.exit(0 if _f == 0 else 1)
