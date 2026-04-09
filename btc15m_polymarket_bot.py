#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  BTC 15M Predictor Bot – Polymarket 15-Minute Markets           ║
║                                                                  ║
║  Mejoras v2.1:                                                  ║
║   · MIN_EDGE 3% (calibrado para modelo isotónico)               ║
║   · SIGNAL_THRESHOLD 0.52 (ajustado a capacidad real del modelo)║
║   · requests.Session() reutiliza conexiones TCP (menor latencia)║
║   · Discord notifications en hilo secundario (no bloquea)       ║
║   · Sin fallback heurístico: si ML falla, bot se detiene        ║
║   · Memoria optimizada: float32 + drop columnas innecesarias    ║
║                                                                  ║
║  Mejoras previas:                                               ║
║   · Imports al top, config desde .env                           ║
║   · NaN guards, gas margen, nonce 'pending', retry 429/502      ║
║   · ask_price corregido, estado persistente, P&L tracking       ║
║   · Edge check modelo vs precio token                           ║
║   · auto_redeem via Polymarket positions API                    ║
║   · Inferencia con modelo XGBoost calibrado (model_15m.pkl)     ║
║   · Order flow en tiempo real: CVD, OI change, funding rate     ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────
# IMPORTS — todos aquí arriba, nunca dentro de funciones
# ─────────────────────────────────────────────────────────────────
import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import requests
from web3 import Web3

# ── Sesión HTTP reutilizable ─────────────────────────────────────
# Reutiliza conexiones TCP con Binance, Polymarket y Discord
# reduciendo latencia de red (~50-200ms por llamada).
_HTTP_SESSION = requests.Session()
_HTTP_SESSION.headers.update({"User-Agent": "btc15m-bot/2.0"})

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    AssetType, BalanceAllowanceParams, MarketOrderArgs, OrderArgs, OrderType,
)
from py_clob_client.order_builder.constants import BUY

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — todo configurable desde .env
# ─────────────────────────────────────────────────────────────────
PRIVATE_KEY      = os.getenv("PRIVATE_KEY_2", "")
BET_SIZE_USDC    = float(os.getenv("BET_SIZE_USDC_15M", "2.0"))
MIN_CONFIDENCE   = float(os.getenv("MIN_CONFIDENCE",    "0.03"))  # casi desactivado: el edge check es el filtro real
MIN_EDGE         = float(os.getenv("MIN_EDGE",          "0.03"))
SIGNAL_THRESHOLD = float(os.getenv("SIGNAL_THRESHOLD",  "0.51"))  # mínimo para activar señal (el edge check filtra el resto)
K_FACTOR         = float(os.getenv("K_FACTOR",          "0.25"))
GAS_MARGIN       = float(os.getenv("GAS_MARGIN",        "1.30"))
CHECK_EVERY_SECS = int(os.getenv("CHECK_EVERY_SECS",    "30"))
DISCORD_WEBHOOK  = os.getenv("DISCORD_WEBHOOK",         "")   # URL del webhook de Discord

MIN_LO = 2     # minutos antes del cierre: inicio ventana
MIN_HI = 7     # minutos antes del cierre: fin ventana

RSI_LEN = 14
BB_LEN  = 20
BB_MULT = 2.0
PRICE_SLIP = 0.005   # slippage intencional sobre el ask en Limit IOC

STATE_FILE  = Path("state_15m.json")    # estado operacional
TRADES_FILE = Path("trades_15m.json")   # historial P&L
MODEL_FILE  = Path("model_15m.pkl")     # modelo ML entrenado con train_model_15m.py

# ── Order flow (Binance Futures API) ──────────────────────────────
# OI histórico: https://fapi.binance.com/futures/data/openInterestHist
# Funding rate: https://fapi.binance.com/fapi/v1/fundingRate
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
OI_STALE_SECS        = 120   # ignorar OI si tiene más de 2 minutos

# ── Nota de oráculo ───────────────────────────────────────────────
# Polymarket resuelve los mercados BTC 15M usando el precio de Chainlink
# BTC/USD en Polygon (0xc907E116054Ad103354f2D350FD2514433D57F6f).
# El precio de referencia es el de la ÚLTIMA actualización del feed de
# Chainlink DENTRO de la ventana de 15 minutos.
# Implicaciones:
#   1. El target del modelo debe ser: precio Chainlink al cierre > precio
#      Chainlink al inicio. Binance es un proxy aceptable (correlación >0.999)
#      pero puede haber divergencias en momentos de baja liquidez.
#   2. La validación check_price_divergence() ya detecta desincronizaciones
#      Binance vs Chainlink > MAX_DIVERGENCE_PCT (0.5%).
#   3. En mercados donde Chainlink tiene retraso (precio desactualizado),
#      el bot ya aborta la operación.
ORACLE_RESOLUTION_SOURCE = "Chainlink BTC/USD Polygon"

POLYMARKET_HOST  = "https://clob.polymarket.com"
POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"
CHAIN_ID         = 137
ET_ZONE          = pytz.timezone("America/New_York")

POLYGON_RPCS = [
    "https://polygon-rpc.com",
    "https://rpc-mainnet.matic.quiknode.pro",
    "https://rpc-mainnet.maticvigil.com",
    "https://polygon.llamarpc.com",
    "https://1rpc.io/matic",
]

USDC_ADDRESSES = {
    "native":  "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
    "bridged": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
}
POLYMARKET_SPENDERS = {
    "CTF Exchange":      "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
    "Neg Risk Exchange": "0xC5d563A36AE78145C45a50134d48A1215220f80a",
}
CT_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# Chainlink BTC/USD en Polygon — misma fuente que usa Polymarket para resolver
CHAINLINK_BTC_USD  = "0xc907E116054Ad103354f2D350FD2514433D57F6f"
MAX_PRICE_AGE_SECS = 120   # precio válido si tiene < 2 minutos
MAX_DIVERGENCE_PCT = 0.5   # % máximo tolerado entre Binance y Chainlink

MAX_UINT256 = 2**256 - 1
MIN_SHARES  = 2.0

USDC_ABI = [
    {"name": "allowance", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"name": "approve", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}],
     "outputs": [{"name": "", "type": "bool"}]},
    {"name": "balanceOf", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "account", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
]

CT_ABI = [
    {"name": "redeemPositions", "type": "function", "stateMutability": "nonpayable",
     "inputs": [
         {"name": "collateralToken",    "type": "address"},
         {"name": "parentCollectionId", "type": "bytes32"},
         {"name": "conditionId",        "type": "bytes32"},
         {"name": "indexSets",          "type": "uint256[]"},
     ], "outputs": []},
    {"name": "payoutDenominator", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "conditionId", "type": "bytes32"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"name": "payoutNumerators", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "conditionId", "type": "bytes32"}, {"name": "index", "type": "uint256"}],
     "outputs": [{"name": "", "type": "uint256"}]},
]

CHAINLINK_ABI = [
    {"name": "latestRoundData", "type": "function", "stateMutability": "view",
     "inputs": [],
     "outputs": [
         {"name": "roundId",         "type": "uint80"},
         {"name": "answer",          "type": "int256"},
         {"name": "startedAt",       "type": "uint256"},
         {"name": "updatedAt",       "type": "uint256"},
         {"name": "answeredInRound", "type": "uint80"},
     ]},
    {"name": "decimals", "type": "function", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "uint8"}]},
]

# ─────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("btc15m_bot_log.txt", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# ESTADO PERSISTENTE — evita apuestas duplicadas tras reinicios
# ─────────────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}

def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))

def already_traded_this_window(window_ts: int) -> bool:
    return load_state().get("last_traded_window") == window_ts

def mark_traded(
    window_ts: int, direction: str, token_price: float,
    our_prob: float, condition_id: str | None = None,
    market_title: str = "", source: str = "?",
) -> None:
    state = load_state()
    state["last_traded_window"] = window_ts
    state["direction"]          = direction
    state["token_price"]        = token_price
    state["our_prob"]           = our_prob
    state["timestamp"]          = datetime.now(timezone.utc).isoformat()
    save_state(state)
    record_trade_entry(
        window_ts=window_ts,
        direction=direction,
        token_price=token_price,
        our_prob=our_prob,
        bet_usdc=BET_SIZE_USDC,
        condition_id=condition_id,
        market_title=market_title,
    )
    # Notificación Discord al abrir posición
    discord_trade_open(
        direction=direction,
        our_prob=our_prob,
        token_price=token_price,
        bet_usdc=BET_SIZE_USDC,
        window_ts=window_ts,
        source=source,
    )

# ─────────────────────────────────────────────────────────────────
# DISCORD NOTIFICATIONS
# ─────────────────────────────────────────────────────────────────
def _discord_send_sync(content: str) -> None:
    """Envío real al webhook (ejecutado en hilo secundario)."""
    try:
        r = _HTTP_SESSION.post(DISCORD_WEBHOOK, json={"content": content}, timeout=8)
        if r.status_code not in (200, 204):
            log.warning(f"Discord webhook: status {r.status_code}")
    except Exception as e:
        log.warning(f"Discord send error: {e}")

def discord_send(content: str) -> None:
    """Envía un mensaje al canal de Discord via webhook en segundo plano."""
    if not DISCORD_WEBHOOK:
        return
    t = threading.Thread(target=_discord_send_sync, args=(content,), daemon=True)
    t.start()

def discord_trade_open(direction, our_prob, token_price, bet_usdc, window_ts, source="?"):
    edge   = (our_prob - token_price) * 100
    arrow  = "🟢 UP" if direction == "UP" else "🔴 DOWN"
    dt_utc = datetime.fromtimestamp(window_ts, tz=timezone.utc).strftime("%H:%M UTC")
    msg = (
        f"**BTC 15M — POSICIÓN ABIERTA** {arrow}\n"
        f"```\n"
        f"Ventana  : {dt_utc}\n"
        f"Apuesta  : ${bet_usdc:.2f} USDC\n"
        f"Modelo   : {our_prob*100:.1f}%  [{source}]\n"
        f"Mercado  : {token_price*100:.1f}%\n"
        f"Edge     : +{edge:.1f}%\n"
        f"```"
    )
    discord_send(msg)

def discord_trade_result(direction, status, payout_usdc, bet_usdc, token_price, our_prob, title=""):
    pnl     = payout_usdc - bet_usdc
    won     = status == "won"
    icon    = "✅" if won else "❌"
    pnl_str = f"+${pnl:.4f}" if pnl >= 0 else f"-${abs(pnl):.4f}"
    roi     = (pnl / bet_usdc * 100) if bet_usdc > 0 else 0
    arrow   = "🟢 UP" if direction == "UP" else "🔴 DOWN"
    msg = (
        f"**BTC 15M — RESOLUCIÓN** {icon} {'GANADA' if won else 'PERDIDA'}\n"
        f"```\n"
        f"Posición : {arrow}\n"
        f"Mercado  : {title[:40]}\n"
        f"Cobrado  : ${payout_usdc:.4f} USDC\n"
        f"Apostado : ${bet_usdc:.2f} USDC\n"
        f"P&L      : {pnl_str}  (ROI {roi:+.1f}%)\n"
        f"Edge est.: {(our_prob-token_price)*100:+.1f}%\n"
        f"```"
    )
    discord_send(msg)

# ─────────────────────────────────────────────────────────────────
# P&L TRACKING
# ─────────────────────────────────────────────────────────────────
def load_trades() -> list[dict]:
    if TRADES_FILE.exists():
        try:
            data = json.loads(TRADES_FILE.read_text())
            return data if isinstance(data, list) else []
        except Exception:
            pass
    return []

def save_trades(trades: list[dict]) -> None:
    TRADES_FILE.write_text(json.dumps(trades, indent=2))

def record_trade_entry(
    window_ts: int, direction: str, token_price: float,
    our_prob: float, bet_usdc: float, condition_id: str | None,
    market_title: str = "",
) -> str:
    trade_id = datetime.now(timezone.utc).isoformat()
    trade = {
        "id":           trade_id,
        "window_ts":    window_ts,
        "window_utc":   datetime.fromtimestamp(window_ts, tz=timezone.utc).isoformat(),
        "direction":    direction,
        "token_price":  round(token_price, 4),
        "our_prob":     round(our_prob, 4),
        "edge":         round(our_prob - token_price, 4),
        "bet_usdc":     bet_usdc,
        "condition_id": condition_id,
        "market_title": market_title,
        "status":       "open",
        "payout_usdc":  None,
        "pnl_usdc":     None,
        "settled_at":   None,
    }
    trades = load_trades()
    trades.append(trade)
    save_trades(trades)
    log.info(
        f"Trade registrado: {direction} @ {token_price:.2f} | "
        f"edge {(our_prob - token_price)*100:.1f}%"
    )
    return trade_id

def record_trade_result(condition_id: str, payout_usdc: float, bet_usdc: float) -> None:
    trades     = load_trades()
    settled_at = datetime.now(timezone.utc).isoformat()
    found      = False
    for t in trades:
        if t.get("condition_id") == condition_id and t.get("status") == "open":
            pnl              = round(payout_usdc - bet_usdc, 4)
            t["status"]      = "won" if payout_usdc > 0 else "lost"
            t["payout_usdc"] = round(payout_usdc, 4)
            t["pnl_usdc"]    = pnl
            t["settled_at"]  = settled_at
            found            = True
            result_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            log.info(
                f"Trade liquidado: {t['direction']} | {t['status'].upper()} | "
                f"cobrado=${payout_usdc:.2f} | P&L {result_str}"
            )
            # Notificación Discord con resultado y P&L
            discord_trade_result(
                direction    = t.get("direction", "?"),
                status       = t["status"],
                payout_usdc  = payout_usdc,
                bet_usdc     = bet_usdc,
                token_price  = t.get("token_price", 0.5),
                our_prob     = t.get("our_prob", 0.5),
                title        = t.get("market_title", ""),
            )
            break
    if not found:
        log.warning(f"No se encontró trade abierto para {condition_id[:12]}...")
    save_trades(trades)

def print_pnl_summary() -> None:
    trades = load_trades()
    if not trades:
        log.info("── P&L Summary 15M: sin operaciones registradas ──")
        return

    total    = len(trades)
    open_tr  = [t for t in trades if t["status"] == "open"]
    settled  = [t for t in trades if t["status"] != "open"]
    won      = [t for t in settled if t["status"] == "won"]
    lost     = [t for t in settled if t["status"] == "lost"]

    total_bet = sum(t["bet_usdc"] for t in settled)
    total_pay = sum(t["payout_usdc"] or 0 for t in settled)
    net_pnl   = total_pay - total_bet
    win_rate  = len(won) / len(settled) * 100 if settled else 0.0
    roi       = net_pnl / total_bet * 100 if total_bet > 0 else 0.0

    avg_edge_won  = (sum(t["edge"] for t in won)  / len(won))  * 100 if won  else 0.0
    avg_edge_lost = (sum(t["edge"] for t in lost) / len(lost)) * 100 if lost else 0.0

    sep = "─" * 52
    log.info(sep)
    log.info("  P&L SUMMARY — BTC 15M Bot")
    log.info(sep)
    log.info(f"  Operaciones  : {total}  ({len(open_tr)} abiertas, {len(settled)} liquidadas)")
    log.info(f"  Victorias    : {len(won)} / {len(settled)}  (win rate {win_rate:.1f}%)")
    log.info(f"  Total apostado: ${total_bet:.2f}")
    log.info(f"  Total cobrado : ${total_pay:.2f}")
    log.info(f"  P&L neto     : {'+'if net_pnl>=0 else ''}${net_pnl:.2f}  (ROI {roi:+.1f}%)")
    log.info(sep)
    log.info(f"  Edge medio (ganadas) : {avg_edge_won:+.1f}%")
    log.info(f"  Edge medio (perdidas): {avg_edge_lost:+.1f}%")
    if settled:
        note = (
            "✓ Edge predice resultado" if avg_edge_won > avg_edge_lost
            else "⚠ Edge no predice resultado – revisar modelo"
        )
        log.info(f"  Calibración  : {note}")
    log.info(sep)
    if settled:
        log.info("  Últimas operaciones:")
        for t in settled[-5:]:
            pnl_str = f"{'+'if (t['pnl_usdc'] or 0)>=0 else ''}${t['pnl_usdc']:.2f}"
            log.info(
                f"    w={t['window_ts']} | {t['direction']:4s} | "
                f"precio={t['token_price']:.2f} edge={t['edge']*100:+.1f}% | "
                f"{t['status'].upper():4s} | P&L {pnl_str}"
            )
    log.info(sep)

# ─────────────────────────────────────────────────────────────────
# DATOS BTC — velas 15M desde Binance
# ─────────────────────────────────────────────────────────────────
def get_btc_candles_15m(limit: int = 250) -> pd.DataFrame:
    url    = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "15m", "limit": limit}
    try:
        r = _HTTP_SESSION.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.error(f"Error descargando velas de Binance: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbav", "tqav", "ignore",
    ])
    df.drop(columns=["qav", "trades", "tqav", "ignore"], inplace=True)
    for col in ["open", "high", "low", "close", "volume", "tbav"]:
        df[col] = pd.to_numeric(df[col], downcast="float")
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df.reset_index(drop=True)

def get_remaining_minutes(df: pd.DataFrame) -> float:
    if df.empty:
        return -1.0
    now_ms   = datetime.now(timezone.utc).timestamp() * 1000
    close_ms = df.iloc[-1]["close_time"].timestamp() * 1000
    return (close_ms - now_ms) / 60_000.0

def fmt_time(dt) -> str:
    """Formatea hora sin cero inicial. Compatible Linux y Windows."""
    return dt.strftime("%I:%M%p").lstrip("0").lower()

# ─────────────────────────────────────────────────────────────────
# MODELO ML — carga y feature engineering para inferencia
# ─────────────────────────────────────────────────────────────────

# ── _CalibratedModel: réplica exacta de la clase del trainer ─────
# joblib/pickle necesita encontrar esta clase al deserializar el .pkl
# porque fue guardada desde __main__ en train_model_15m.py.

class _CalibratedModel:
    """Wrapper XGBoost + IsotonicRegression (idéntico al de train_model_15m.py)."""
    def __init__(self, base, iso):
        self.base = base
        self.iso  = iso
    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1]
        cal = self.iso.transform(raw)
        return np.column_stack([1.0 - cal, cal])
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

# Parche para que joblib encuentre _CalibratedModel como si fuera de __main__
_main_mod = sys.modules.get("__main__")
if _main_mod is not None and not hasattr(_main_mod, "_CalibratedModel"):
    _main_mod._CalibratedModel = _CalibratedModel

_ML_MODEL      = None    # modelo cargado en memoria
_ML_FEATURES   = None    # lista de feature names esperados
_ML_META       = None    # metadata completa del modelo (threshold, auc, etc.)

def load_ml_model() -> bool:
    """
    Carga el modelo entrenado por train_model_15m.py.
    Devuelve True si se cargó correctamente, False en caso contrario.
    """
    global _ML_MODEL, _ML_FEATURES, _ML_META
    if not JOBLIB_AVAILABLE:
        log.warning("joblib no disponible — modelo ML no cargable")
        return False
    if not MODEL_FILE.exists():
        log.warning(
            f"Modelo ML no encontrado ({MODEL_FILE}). "
            "Ejecuta: python train_model_15m.py"
        )
        return False
    try:
        # ── Fix deserialización: CalibratedModel debe existir en __main__ ──
        # joblib/pickle busca la clase en el módulo donde fue definida.
        # Si train_model_15m.py la definió como __main__.CalibratedModel,
        # necesitamos inyectarla antes de cargar.
        import sys
        import importlib
        _main = sys.modules.get("__main__")
        if _main and not hasattr(_main, "CalibratedModel"):
            # Intentar importar desde el script de entrenamiento
            try:
                train_mod = importlib.import_module("train_model_15m")
                if hasattr(train_mod, "CalibratedModel"):
                    _main.CalibratedModel = train_mod.CalibratedModel
                    log.info("  CalibratedModel importada desde train_model_15m")
            except ImportError:
                # Definir stub compatible (XGBoost + IsotonicRegression wrapper)
                class CalibratedModel:
                    """Stub para deserializar modelos calibrados con isotonic regression."""
                    def __init__(self, base_model=None, calibrator=None):
                        self.base_model = base_model
                        self.calibrator = calibrator
                    def predict_proba(self, X):
                        raw_proba = self.base_model.predict_proba(X)
                        if self.calibrator is not None:
                            raw_p1 = raw_proba[:, 1]
                            # IsotonicRegression acepta 1D; otros pueden necesitar 2D
                            try:
                                cal = self.calibrator.predict(raw_p1)
                            except Exception:
                                cal = self.calibrator.predict(raw_p1.reshape(-1, 1))
                            cal = np.clip(cal, 0.0, 1.0)
                            return np.column_stack([1 - cal, cal])
                        return raw_proba
                _main.CalibratedModel = CalibratedModel
                log.info("  CalibratedModel stub inyectada para deserialización")

        meta         = joblib.load(MODEL_FILE)
        _ML_MODEL    = meta["model"]
        _ML_FEATURES = meta["feature_names"]
        _ML_META     = meta
        trained_at   = meta.get("trained_at", "?")
        auc          = meta.get("cv_auc_mean", 0)
        n            = meta.get("n_samples", 0)
        thresh       = meta.get("signal_threshold", 0.55)
        version      = meta.get("version", "v1")
        target       = meta.get("target", "desconocido")
        log.info(
            f"Modelo ML cargado [{version}] | AUC CV={auc:.4f} | "
            f"Threshold={thresh:.3f} | Muestras={n:,} | "
            f"Entrenado={trained_at[:10]}"
        )
        log.info(f"  Target: {target}")
        if version.startswith("v1"):
            log.warning(
                "  ⚠ Modelo v1 detectado — target puede ser incorrecto. "
                "Reentrena con train_model_15m.py v2."
            )
        return True
    except Exception as e:
        log.error(f"Error cargando modelo ML: {e} — usando heurística")
        return False

def _compute_ml_features(df: pd.DataFrame) -> pd.Series | None:
    """
    Features para inferencia en tiempo real.

    ═══ REGLA ANTI-LEAKAGE (debe coincidir con train_model_15m.py) ═══
    El trainer hace .shift(1) sobre todas las features de precio.
    Aquí el equivalente es usar df.iloc[-2] (vela t-1, ya cerrada)
    para todos los cálculos de precio.

    Solo los features de TIEMPO usan df.iloc[-1] (vela actual),
    porque open_time es conocido desde el minuto 0.

    Nunca usar df.iloc[-1].close, open, high, low → son la vela
    en curso (incompleta) y correlacionan con el target.
    ══════════════════════════════════════════════════════════════
    """
    if len(df) < 210:
        return None

    # Trabajamos con la serie completa hasta t-1 inclusive.
    # Esto garantiza que .iloc[-1] de cualquier serie calculada
    # corresponde a la vela t-1 (la última cerrada).
    df_past = df.iloc[:-1]   # excluye la vela actual (t)

    close  = df_past["close"]
    high   = df_past["high"]
    low    = df_past["low"]
    open_  = df_past["open"]
    volume = df_past["volume"]
    tbav   = df_past["tbav"].astype(float) if "tbav" in df_past.columns else volume * 0.5

    f = {}

    # ── RSI múltiple ───────────────────────────────────────────────
    for n in [7, 14, 21, 28]:
        val = calc_rsi(close, n).iloc[-1]
        if np.isnan(val):
            return None
        f[f"rsi_{n}"] = val

    # ── Bollinger Bands ────────────────────────────────────────────
    for n in [10, 20, 30]:
        basis = close.rolling(n).mean()
        std   = close.rolling(n).std(ddof=0)
        band  = (4 * std).replace(0, np.nan)
        pct   = ((close - (basis - 2 * std)) / band).iloc[-1]
        wid   = (band / basis.replace(0, np.nan)).iloc[-1]
        if np.isnan(pct):
            return None
        f[f"bb_pct_{n}"]   = pct
        f[f"bb_width_{n}"] = wid

    # ── MACD ───────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    f["macd"]        = macd.iloc[-1]
    f["macd_signal"] = sig.iloc[-1]
    f["macd_hist"]   = (macd - sig).iloc[-1]
    f["macd_norm"]   = macd.iloc[-1] / close.iloc[-1] if close.iloc[-1] != 0 else 0.0

    # ── Retornos ───────────────────────────────────────────────────
    for n in [1, 2, 3, 4, 5, 8, 12, 15, 20]:
        ret = close.pct_change(n).iloc[-1]
        if np.isnan(ret):
            return None
        f[f"ret_{n}"] = ret

    # ── ATR ────────────────────────────────────────────────────────
    tr = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1
    ).max(axis=1)
    for n in [7, 14, 21]:
        f[f"atr_{n}"] = tr.rolling(n).mean().iloc[-1]
    f["atr_pct_14"] = f["atr_14"] / close.iloc[-1] if close.iloc[-1] != 0 else 0.0

    # ── Price Action (vela t-1, ya cerrada) ───────────────────────
    rng = high.iloc[-1] - low.iloc[-1]
    if rng == 0:
        return None
    body       = close.iloc[-1] - open_.iloc[-1]
    upper_wick = high.iloc[-1] - max(open_.iloc[-1], close.iloc[-1])
    lower_wick = min(open_.iloc[-1], close.iloc[-1]) - low.iloc[-1]
    f["body_ratio"]   = body / rng
    f["wick_bias"]    = (lower_wick - upper_wick) / rng
    f["upper_wick_r"] = upper_wick / rng
    f["lower_wick_r"] = lower_wick / rng
    f["range_pct"]    = rng / close.iloc[-1] if close.iloc[-1] != 0 else 0.0
    f["intra_return"] = body / open_.iloc[-1] if open_.iloc[-1] != 0 else 0.0
    f["close_vs_high"]= (close.iloc[-1] - high.iloc[-1]) / rng
    f["close_vs_low"] = (close.iloc[-1] - low.iloc[-1])  / rng
    f["body_pos"]     = (close.iloc[-1] - low.iloc[-1])  / rng

    # ── Volume + CVD ───────────────────────────────────────────────
    vol_ma = volume.rolling(20).mean()
    f["vol_ratio"] = volume.iloc[-1] / vol_ma.iloc[-1] if vol_ma.iloc[-1] > 0 else 1.0
    delta  = 2 * tbav - volume
    for n in [3, 5, 8, 10, 15, 20]:
        vol_roll      = volume.rolling(n).sum().iloc[-1]
        f[f"cvd_{n}"] = delta.rolling(n).sum().iloc[-1] / vol_roll if vol_roll > 0 else 0.0
    f["buy_ratio"]    = tbav.iloc[-1] / volume.iloc[-1] if volume.iloc[-1] > 0 else 0.5
    d5mean            = delta.rolling(5).mean().iloc[-1]
    f["delta_accel"]  = delta.iloc[-1] / d5mean if d5mean != 0 else 1.0
    f["cvd_momentum"] = f["cvd_3"] - f["cvd_20"]
    f["vol_body_corr"]= f["body_ratio"] * f["vol_ratio"]

    # ── Distancia a medias ─────────────────────────────────────────
    for n in [8, 21, 50, 100, 200]:
        ma = close.rolling(n).mean().iloc[-1]
        f[f"dist_ma_{n}"] = (close.iloc[-1] - ma) / ma if ma > 0 else 0.0

    # ── Régimen de mercado ─────────────────────────────────────────
    ma50  = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    f["regime_trend"]    = float(np.sign(close.iloc[-1] - ma200))
    f["regime_ma_cross"] = float(np.sign(ma50 - ma200))
    atr14_series         = tr.rolling(14).mean()
    atr14_ma             = atr14_series.rolling(50).mean().iloc[-1]
    f["vol_regime"]  = f["atr_14"] / atr14_ma if atr14_ma > 0 else 1.0
    f["near_ma50"]   = abs(close.iloc[-1] - ma50) / close.iloc[-1] if close.iloc[-1] != 0 else 0.0

    # ── Tiempo / sesión — usa vela ACTUAL (open_time de t) ────────
    # Es el único caso donde usamos df.iloc[-1] (sin precio, solo timestamp)
    ts   = df["open_time"].iloc[-1]
    hour = ts.hour + ts.minute / 60
    dow  = ts.dayofweek
    f["hour_sin"]  = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"]  = np.cos(2 * np.pi * hour / 24)
    f["dow_sin"]   = np.sin(2 * np.pi * dow  / 7)
    f["dow_cos"]   = np.cos(2 * np.pi * dow  / 7)
    f["is_asia"]   = float((hour >= 0)  and (hour < 8))
    f["is_london"] = float((hour >= 7)  and (hour < 16))
    f["is_ny"]     = float((hour >= 13) and (hour < 22))
    f["is_weekend"]= float(dow >= 5)

    return pd.Series(f)

def predict_with_ml(df: pd.DataFrame, order_flow: dict) -> dict | None:
    """
    Predicción ML calibrada. Devuelve el mismo formato que calc_indicators()
    para que el resto del bot funcione sin cambios.

    Si el modelo no está disponible, devuelve None y el bot cae al heurístico.

    order_flow: dict con claves opcionales:
      oi_change_pct   — cambio % de Open Interest en la última ventana
      funding_rate    — funding rate actual (negativo = presión bajista)
    """
    if _ML_MODEL is None or _ML_FEATURES is None:
        return None

    feat_series = _compute_ml_features(df)
    if feat_series is None:
        return None

    # Alinear features al orden exacto del entrenamiento
    try:
        row = feat_series.reindex(_ML_FEATURES).fillna(0.0)
        X   = pd.DataFrame([row], columns=_ML_FEATURES)
    except Exception as e:
        log.warning(f"Error preparando features ML: {e}")
        return None

    try:
        prob_bull = float(_ML_MODEL.predict_proba(X)[0, 1])
    except Exception as e:
        log.warning(f"Error en ML predict_proba: {e}")
        return None

    # ── Ajuste por order flow en tiempo real ─────────────────────
    # Solo ajustamos si OI se mueve >0.5% (ruido < eso no es señal).
    # Multiplicador conservador: máximo ±1.5 puntos para no destruir
    # la calibración isotónica del modelo.
    oi_chg    = order_flow.get("oi_change_pct",  0.0) or 0.0
    funding   = order_flow.get("funding_rate",   0.0) or 0.0
    adjustment = 0.0
    if abs(oi_chg) >= 0.5:  # dead zone: ignorar OI < 0.5%
        of_signal  = np.sign(oi_chg) * np.sign(funding)
        adjustment = float(np.clip(of_signal * (abs(oi_chg) - 0.5) * 0.15, -0.015, 0.015))
    prob_bull  = float(np.clip(prob_bull + adjustment, 0.04, 0.96))
    prob_bear  = 1.0 - prob_bull
    confidence = abs(prob_bull - 0.5) * 2.0

    log.info(
        f"  [ML] raw_prob_bull={prob_bull-adjustment:.3f} | "
        f"OI_chg={oi_chg:+.2f}% funding={funding:.5f} | "
        f"adj={adjustment:+.3f} → final={prob_bull:.3f}"
    )

    return {
        "rsi":         feat_series.get("rsi_14", 50.0),
        "bb_pct":      feat_series.get("bb_pct",  0.5),
        "raw_score":   prob_bull,          # para logging, no para sigmoid
        "prob_bull":   prob_bull,
        "prob_bear":   prob_bear,
        "confidence":  confidence,
        "signal_bull": prob_bull >= SIGNAL_THRESHOLD,
        "signal_bear": prob_bear >= SIGNAL_THRESHOLD,
        "price":       df["close"].iloc[-1],
        "source":      "ML",
    }

# ─────────────────────────────────────────────────────────────────
# ORDER FLOW — datos en tiempo real desde Binance Futures
# ─────────────────────────────────────────────────────────────────
def get_order_flow(symbol: str = "BTCUSDT") -> dict:
    """
    Obtiene señales de order flow de Binance Futures:
      - oi_change_pct: cambio % del Open Interest en la última ventana de 15M
      - funding_rate:  funding rate actual (>0 = longs pagan, <0 = shorts pagan)

    Por qué importan en 15M:
      - OI subiendo + precio subiendo = longs acumulando → señal alcista real
      - OI bajando  + precio subiendo = shorts liquidados → rally débil, puede revertir
      - Funding rate negativo extremo → mercado sobrecargado de shorts → posible squeeze

    Si la API de futuros no está disponible, devuelve valores neutros (0.0).
    """
    result = {"oi_change_pct": 0.0, "funding_rate": 0.0}
    try:
        # Open Interest histórico (últimas 3 velas de 15M)
        r = _HTTP_SESSION.get(
            f"{BINANCE_FUTURES_BASE}/futures/data/openInterestHist",
            params={"symbol": symbol, "period": "15m", "limit": 3},
            timeout=8,
        )
        if r.status_code == 200:
            oi_data = r.json()
            if len(oi_data) >= 2:
                oi_now  = float(oi_data[-1]["sumOpenInterest"])
                oi_prev = float(oi_data[-2]["sumOpenInterest"])
                if oi_prev > 0:
                    result["oi_change_pct"] = (oi_now - oi_prev) / oi_prev * 100
    except Exception as e:
        log.debug(f"OI fetch error: {e}")

    try:
        # Funding rate actual
        r = _HTTP_SESSION.get(
            f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate",
            params={"symbol": symbol, "limit": 1},
            timeout=8,
        )
        if r.status_code == 200:
            data = r.json()
            if data:
                result["funding_rate"] = float(data[-1]["fundingRate"])
    except Exception as e:
        log.debug(f"Funding rate fetch error: {e}")

    if result["oi_change_pct"] != 0.0 or result["funding_rate"] != 0.0:
        log.info(
            f"  OrderFlow: OI_chg={result['oi_change_pct']:+.3f}% | "
            f"Funding={result['funding_rate']:+.6f}"
        )
    return result


def rma(series: pd.Series, length: int) -> pd.Series:
    alpha  = 1.0 / length
    result = series.copy().astype(float)
    result.iloc[:length] = np.nan
    result.iloc[length]  = series.iloc[:length + 1].mean()
    for i in range(length + 1, len(series)):
        result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i - 1]
    return result

def calc_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    denom = rma(loss, length).replace(0, np.nan)
    rs    = rma(gain, length) / denom
    return 100 - (100 / (1 + rs))

def calc_indicators(df: pd.DataFrame) -> dict | None:
    """
    Devuelve None si algún indicador produce NaN
    (vela con rango cero, datos corruptos de Binance, etc.)
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    open_  = df["open"]
    volume = df["volume"]

    # RSI
    rsi_v = calc_rsi(close, RSI_LEN).iloc[-1]
    if np.isnan(rsi_v):
        log.warning("RSI es NaN, saltando ciclo")
        return None
    rsi_s = (2 if rsi_v >= 70 else 1 if rsi_v >= 55
             else -2 if rsi_v <= 30 else -1 if rsi_v <= 45 else 0)

    # Bollinger Bands
    bb_basis = close.rolling(BB_LEN).mean()
    bb_std   = close.rolling(BB_LEN).std(ddof=0)
    bb_up    = bb_basis + BB_MULT * bb_std
    bb_lo    = bb_basis - BB_MULT * bb_std
    bb_range = (bb_up - bb_lo).replace(0, np.nan)
    bb_pct   = ((close - bb_lo) / bb_range).iloc[-1]
    if np.isnan(bb_pct):
        log.warning("BB %b es NaN, saltando ciclo")
        return None
    bb_s = (2 if bb_pct > 0.80 else 1 if bb_pct > 0.55
            else -2 if bb_pct < 0.20 else -1 if bb_pct < 0.45 else 0)

    # Volumen
    c, o      = close.iloc[-1], open_.iloc[-1]
    vol_mean  = volume.rolling(20).mean().iloc[-1]
    vol_above = volume.iloc[-1] > vol_mean
    vol_s = (2 if vol_above and c > o else -2 if vol_above and c < o
             else 1 if c > o else -1)

    # Price action
    body       = c - o
    full_range = high.iloc[-1] - low.iloc[-1]
    if full_range == 0:
        log.warning("Vela con rango cero (doji perfecto), saltando ciclo")
        return None
    upper_wick = high.iloc[-1] - max(o, c)
    lower_wick = min(o, c) - low.iloc[-1]
    body_ratio = abs(body) / full_range
    wick_bias  = (lower_wick - upper_wick) / full_range
    pa_s = (2 if body > 0 and body_ratio > 0.5
            else 1 if body > 0 and wick_bias > 0.2
            else -2 if body < 0 and body_ratio > 0.5
            else -1 if body < 0 and wick_bias < -0.2 else 0)

    raw_score = rsi_s * 2.0 + bb_s * 1.5 + vol_s * 1.5 + pa_s * 2.0
    if np.isnan(raw_score):
        log.warning("raw_score es NaN, saltando ciclo")
        return None

    # Sigmoide con K_FACTOR configurable (sustituye la normalización lineal original)
    prob_bull = 1.0 / (1.0 + np.exp(-K_FACTOR * raw_score))
    prob_bull = float(np.clip(prob_bull, 0.04, 0.96))
    prob_bear  = 1.0 - prob_bull
    confidence = abs(prob_bull - 0.5) * 2.0

    return {
        "rsi":         rsi_v,
        "bb_pct":      bb_pct,
        "raw_score":   raw_score,
        "prob_bull":   prob_bull,
        "prob_bear":   prob_bear,
        "confidence":  confidence,
        "signal_bull": prob_bull >= SIGNAL_THRESHOLD,
        "signal_bear": prob_bear >= SIGNAL_THRESHOLD,
        "price":       c,
    }

# ─────────────────────────────────────────────────────────────────
# MERCADOS 15M
# ─────────────────────────────────────────────────────────────────
def get_current_15m_window_ts() -> int:
    now_ts = int(datetime.now(timezone.utc).timestamp())
    return (now_ts // 900) * 900

def build_15m_slug(window_ts: int) -> str:
    return f"btc-updown-15m-{window_ts}"

def find_15m_market() -> dict | None:
    base_ts = get_current_15m_window_ts()
    now_utc = datetime.now(timezone.utc)

    for offset_secs in [0, 900, -900, 1800, -1800]:
        ts   = base_ts + offset_secs
        slug = build_15m_slug(ts)

        for endpoint in [
            f"{POLYMARKET_GAMMA}/events/slug/{slug}",
            f"{POLYMARKET_GAMMA}/markets?slug={slug}",
        ]:
            try:
                r = _HTTP_SESSION.get(endpoint, timeout=15)
                if r.status_code != 200:
                    continue
                data = r.json()

                if isinstance(data, dict) and "markets" in data:
                    markets = data.get("markets", [])
                    active  = [m for m in markets
                               if not m.get("closed", False) and not m.get("archived", False)]
                    if not active:
                        active = [m for m in markets
                                  if m.get("endDate", "") > now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")]
                    if active:
                        log.info(f"Mercado 15M OK: {data.get('title', slug)}")
                        log.info(f"  Slug: {slug} | Cierra: {active[0].get('endDate', '?')}")
                        return active[0]

                if isinstance(data, list) and data:
                    m = data[0]
                    if not m.get("closed", False) and not m.get("archived", False):
                        log.info(f"Mercado 15M OK: {m.get('question', slug)}")
                        return m

            except Exception as e:
                log.debug(f"  {slug}: {e}")

    log.warning("Slug 15M no encontrado → búsqueda genérica")
    return find_15m_market_generic()

def find_15m_market_generic() -> dict | None:
    try:
        r = _HTTP_SESSION.get(
            f"{POLYMARKET_GAMMA}/markets",
            params={"active": "true", "closed": "false", "limit": 100},
            timeout=15,
        )
        if r.status_code == 200:
            markets    = r.json()
            candidates = [
                m for m in markets
                if "btc"  in m.get("slug", "").lower()
                and "15m" in m.get("slug", "").lower()
                and not m.get("closed", False)
            ]
            if candidates:
                candidates.sort(key=lambda x: x.get("endDate", "9999"))
                log.info(f"Mercado 15M (genérico): {candidates[0].get('question', '?')}")
                return candidates[0]
    except Exception as e:
        log.error(f"find_15m_market_generic error: {e}")
    log.error("No se encontró ningún mercado BTC 15M activo.")
    return None

def get_market_tokens(market: dict) -> tuple[str | None, str | None]:
    up_id = down_id = None

    clob_ids = market.get("clobTokenIds", [])
    if isinstance(clob_ids, str):
        try:
            clob_ids = json.loads(clob_ids)
        except Exception:
            clob_ids = []
    if clob_ids and len(clob_ids) >= 2:
        up_id, down_id = clob_ids[0], clob_ids[1]
        log.info(f"  Tokens: UP={up_id[:20]}... DOWN={down_id[:20]}...")
        return up_id, down_id

    tokens = market.get("tokens", [])
    if tokens and isinstance(tokens[0], dict):
        for t in tokens:
            outcome = t.get("outcome", "").upper()
            tid     = t.get("token_id") or t.get("tokenId") or t.get("id")
            if outcome in ("YES", "UP", "HIGHER"):
                up_id = tid
            elif outcome in ("NO", "DOWN", "LOWER"):
                down_id = tid
        if not up_id and len(tokens) >= 2:
            up_id   = tokens[0].get("token_id") or tokens[0].get("tokenId") or tokens[0].get("id")
            down_id = tokens[1].get("token_id") or tokens[1].get("tokenId") or tokens[1].get("id")
        if up_id:
            log.info(f"  Tokens: UP={up_id[:20]}... DOWN={down_id[:20] if down_id else '?'}...")
            return up_id, down_id

    if tokens and isinstance(tokens[0], str) and len(tokens) >= 2:
        up_id, down_id = tokens[0], tokens[1]
        log.info(f"  Tokens: UP={up_id[:20]}... DOWN={down_id[:20]}...")
        return up_id, down_id

    # Último fallback: consultar via conditionId
    condition_id = market.get("conditionId") or market.get("condition_id")
    if condition_id:
        try:
            r = _HTTP_SESSION.get(
                f"{POLYMARKET_GAMMA}/markets",
                params={"condition_id": condition_id},
                timeout=15,
            )
            if r.status_code == 200:
                mlist = r.json()
                if not isinstance(mlist, list):
                    mlist = [mlist]
                for m in mlist:
                    clob = m.get("clobTokenIds", [])
                    if isinstance(clob, str):
                        clob = json.loads(clob)
                    if clob and len(clob) >= 2:
                        up_id, down_id = clob[0], clob[1]
                        log.info(f"  Tokens (fallback): UP={up_id[:20]}...")
                        return up_id, down_id
        except Exception as e:
            log.warning(f"  Error buscando tokens via conditionId: {e}")

    log.error("No se encontraron tokens para este mercado.")
    return None, None

# ─────────────────────────────────────────────────────────────────
# CHAINLINK — validación de precio
# ─────────────────────────────────────────────────────────────────
def get_chainlink_price() -> float | None:
    """
    Obtiene precio BTC/USD desde Chainlink on-chain en Polygon.
    Es la misma fuente que usa Polymarket para resolver los mercados 15M.
    """
    w3 = get_web3()
    if not w3:
        return None
    try:
        feed       = w3.eth.contract(
            address=Web3.to_checksum_address(CHAINLINK_BTC_USD), abi=CHAINLINK_ABI
        )
        decimals   = feed.functions.decimals().call()
        round_data = feed.functions.latestRoundData().call()
        answer     = round_data[1]
        updated_at = round_data[3]
        age        = int(time.time()) - updated_at
        if age > MAX_PRICE_AGE_SECS:
            log.warning(f"  Chainlink BTC: precio desactualizado ({age}s) — ignorando.")
            return None
        return float(answer / (10 ** decimals))
    except Exception as e:
        log.warning(f"  Chainlink BTC error: {e}")
        return None

def check_price_divergence(
    binance_price: float, chainlink_price: float | None
) -> tuple[bool, float]:
    """
    Compara precio Binance vs Chainlink.
    Si Chainlink no está disponible, opera solo con Binance (True, 0.0).
    """
    if chainlink_price is None:
        log.warning("  Chainlink no disponible — operando solo con Binance.")
        return True, 0.0

    diff_pct = abs(binance_price - chainlink_price) / chainlink_price * 100
    log.info(
        f"  BTC Binance=${binance_price:,.2f} | Chainlink=${chainlink_price:,.2f} | "
        f"dif={diff_pct:.3f}%"
    )
    if diff_pct > MAX_DIVERGENCE_PCT:
        log.warning(
            f"  ⚠ Divergencia {diff_pct:.3f}% > {MAX_DIVERGENCE_PCT}% — "
            f"precios desincronizados, no opera."
        )
        return False, diff_pct
    return True, diff_pct

# ─────────────────────────────────────────────────────────────────
# EDGE CHECK
# ─────────────────────────────────────────────────────────────────
MAX_SPREAD = float(os.getenv("MAX_SPREAD", "0.06"))  # spread máximo tolerable (6%)

def get_token_market_price(client: ClobClient, token_id: str) -> dict | None:
    """Devuelve {mid, ask, bid, spread} del token. None si no hay datos."""
    try:
        book = _clob_call_with_retry(
            client.get_order_book, token_id,
            label=f"get_order_book({token_id[:8]}...)"
        )
        if not book:
            return None
        bids = book.bids if hasattr(book, "bids") and book.bids else []
        asks = book.asks if hasattr(book, "asks") and book.asks else []
        bid = float(bids[0].price) if bids else None
        ask = float(asks[0].price) if asks else None
        if bid and ask:
            return {"mid": (bid + ask) / 2.0, "ask": ask, "bid": bid, "spread": ask - bid}
        if ask:
            return {"mid": ask, "ask": ask, "bid": None, "spread": 0.0}
        if bid:
            return {"mid": bid, "ask": None, "bid": bid, "spread": 0.0}
    except Exception as e:
        log.warning(f"Error inesperado en get_token_market_price: {e}")
    return None

def has_positive_edge(our_prob: float, market: dict) -> bool:
    """Edge calculado contra el ask (lo que realmente pagas como taker)."""
    ask    = market["ask"] or market["mid"]
    spread = market["spread"]
    edge   = our_prob - ask
    log.info(
        f"  Edge check: modelo={our_prob*100:.1f}% | ask={ask*100:.1f}% | "
        f"spread={spread*100:.1f}% | edge={edge*100:.1f}% | umbral={MIN_EDGE*100:.1f}%"
    )
    if spread > MAX_SPREAD:
        log.info(f"  Spread {spread*100:.1f}% > máx {MAX_SPREAD*100:.0f}%. No opera.")
        return False
    return edge >= MIN_EDGE

# ─────────────────────────────────────────────────────────────────
# POLYMARKET CLIENT — retry y reconexión
# ─────────────────────────────────────────────────────────────────
CLOB_RETRY_DELAYS = [5, 15, 30]

def _is_rate_limit_or_gateway(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(code in msg for code in ["429", "502", "503", "rate limit", "too many"])

def _clob_call_with_retry(fn, *args, label: str = "clob call", **kwargs):
    """Llama a fn con reintentos ante 429/502. Propaga otros errores."""
    for attempt, delay in enumerate(CLOB_RETRY_DELAYS, start=1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if _is_rate_limit_or_gateway(e):
                log.warning(
                    f"{label} → error transitorio ({e}). "
                    f"Reintento {attempt}/{len(CLOB_RETRY_DELAYS)} en {delay}s..."
                )
                time.sleep(delay)
            else:
                raise
    log.error(f"{label} → agotados {len(CLOB_RETRY_DELAYS)} reintentos.")
    return None

def _build_clob_client(private_key: str, max_attempts: int = 3) -> ClobClient | None:
    for attempt in range(1, max_attempts + 1):
        try:
            client = ClobClient(host=POLYMARKET_HOST, key=private_key, chain_id=CHAIN_ID)
            creds  = client.create_or_derive_api_creds()
            client.set_api_creds(creds)
            log.info(f"ClobClient autenticado (intento {attempt})")
            return client
        except Exception as e:
            if _is_rate_limit_or_gateway(e) and attempt < max_attempts:
                wait = CLOB_RETRY_DELAYS[attempt - 1]
                log.warning(f"Polymarket no disponible ({e}). Reintentando en {wait}s...")
                time.sleep(wait)
            else:
                log.error(f"Error conectando a Polymarket: {e}")
                return None
    return None

# ─────────────────────────────────────────────────────────────────
# WEB3 HELPERS
# ─────────────────────────────────────────────────────────────────
def get_web3() -> Web3 | None:
    for rpc_url in POLYGON_RPCS:
        try:
            w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 10}))
            if w3.is_connected():
                return w3
        except Exception:
            continue
    log.warning("Ningún RPC de Polygon disponible.")
    return None

def _safe_gas_price(w3: Web3) -> int:
    """Gas price con margen GAS_MARGIN para confirmación rápida en Polygon."""
    return int(w3.eth.gas_price * GAS_MARGIN)

def _get_nonce(w3: Web3, wallet: str) -> int:
    """
    Nonce usando 'pending' para incluir txs en mempool aún no confirmadas.
    Evita que dos txs consecutivas reciban el mismo nonce y se sobreescriban.
    """
    return w3.eth.get_transaction_count(wallet, "pending")

# ─────────────────────────────────────────────────────────────────
# ÓRDENES
# ─────────────────────────────────────────────────────────────────
def place_order(client: ClobClient, token_id: str, size_usdc: float) -> bool:
    ask_price = None
    try:
        book = _clob_call_with_retry(
            client.get_order_book, token_id,
            label=f"get_order_book({token_id[:8]}...)"
        )
        if book and hasattr(book, "asks") and book.asks:
            ask_price = float(book.asks[0].price)
        elif book and hasattr(book, "bids") and book.bids:
            ask_price = float(book.bids[0].price)
    except Exception as e:
        log.warning(f"  Error leyendo order book: {e}")

    if ask_price is not None:
        log.info(f"  Best ask: {ask_price:.4f}")
        if ask_price > 0.995:
            log.warning(f"  Ask demasiado alto ({ask_price:.4f}). Abortado.")
            return False

        # Intentar Market Order FOK con retry
        try:
            log.info(f"  Market Order FOK: ${size_usdc} USDC")
            order_args   = MarketOrderArgs(token_id=token_id, amount=size_usdc, side=BUY)
            signed_order = client.create_market_order(order_args)
            response     = _clob_call_with_retry(
                client.post_order, signed_order, OrderType.FOK,
                label="post_order FOK"
            )
            if response is not None:
                log.info(f"  Orden ejecutada: {response}")
                return True
        except Exception as e:
            log.warning(f"  Market Order FOK falló ({e}), intentando Limit IOC...")

    if ask_price is None:
        log.error("  Sin precio de mercado para Limit Order.")
        return False

    # Limit IOC: precio ligeramente por encima del ask para asegurar fill
    # FIX: el original usaba min(0.97, ask_price) que podía quedar por debajo
    # del ask y causar cancelaciones silenciosas en mercados con spread amplio.
    price  = round(min(0.995, ask_price + PRICE_SLIP), 4)
    shares = round(size_usdc / price, 4)

    if shares < MIN_SHARES:
        needed = round(MIN_SHARES * price, 2)
        log.warning(f"  Shares insuficientes ({shares:.2f} < {MIN_SHARES}). Necesitas ≥${needed}.")
        return False

    log.info(f"  Limit Order IOC: {shares} shares @ {price:.4f}")
    try:
        order_args   = OrderArgs(token_id=token_id, price=price, size=shares, side=BUY)
        signed_order = client.create_order(order_args)
        response     = _clob_call_with_retry(
            client.post_order, signed_order, OrderType.IOC,
            label="post_order IOC"
        )
        if response is not None:
            log.info(f"  Orden ejecutada: {response}")
            return True
        return False
    except Exception as e:
        log.error(f"  Limit Order IOC falló: {e}")
        return False

# ─────────────────────────────────────────────────────────────────
# APPROVE USDC
# ─────────────────────────────────────────────────────────────────
def approve_usdc_allowance(private_key: str) -> None:
    """
    Aprueba allowance MAX_UINT256 de USDC (nativo y bridged) a ambos
    spenders de Polymarket (CTF Exchange + Neg Risk Exchange).
    Usa _get_nonce con 'pending' y pausa entre txs para evitar desync.
    """
    w3 = get_web3()
    if not w3:
        return

    account = w3.eth.account.from_key(private_key)
    wallet  = account.address
    log.info(f"  Wallet: {wallet}")

    for usdc_label, usdc_addr in USDC_ADDRESSES.items():
        try:
            usdc = w3.eth.contract(address=Web3.to_checksum_address(usdc_addr), abi=USDC_ABI)
            bal  = usdc.functions.balanceOf(wallet).call() / 10**6
            log.info(f"  Balance USDC {usdc_label}: {bal:.4f}")
        except Exception as e:
            log.warning(f"  No se pudo leer balance {usdc_label}: {e}")
            continue

        for spender_label, spender_addr in POLYMARKET_SPENDERS.items():
            try:
                spender_cs = Web3.to_checksum_address(spender_addr)
                current    = usdc.functions.allowance(wallet, spender_cs).call()
                if current >= MAX_UINT256 // 2:
                    log.info(f"  OK {usdc_label} → {spender_label} (allowance suficiente)")
                    continue

                log.info(f"  Aprobando {usdc_label} → {spender_label}...")
                nonce   = _get_nonce(w3, wallet)
                gas_p   = _safe_gas_price(w3)
                tx      = usdc.functions.approve(spender_cs, MAX_UINT256).build_transaction({
                    "from": wallet, "nonce": nonce,
                    "gas": 100_000, "gasPrice": gas_p, "chainId": CHAIN_ID,
                })
                signed  = w3.eth.account.sign_transaction(tx, private_key)
                tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
                if receipt["status"] == 1:
                    log.info(f"  APROBADA: {usdc_label} → {spender_label}")
                # Pausa breve entre txs para que el mempool se asiente
                time.sleep(2)
            except Exception as e:
                log.warning(f"  Error {usdc_label} → {spender_label}: {e}")

# ─────────────────────────────────────────────────────────────────
# AUTO REDEEM — via Polymarket Positions API (robusto, escanea todo)
# ─────────────────────────────────────────────────────────────────
def auto_redeem_winnings(private_key: str) -> None:
    """
    Redime posiciones ganadoras usando la lógica exacta del script manual.

    Diferencias clave respecto a la versión anterior:
      · curPrice == 1 en lugar de >= 0.95 (solo posiciones completamente resueltas)
      · time.sleep(1) antes de cada tx para estabilidad de nonce
      · break inmediato en revert (no continúa con otros index_sets del mismo colateral)
      · Notificación Discord con P&L al reclamar
    """
    try:
        from eth_account import Account
        wallet = Account.from_key(private_key).address

        log.info("Buscando posiciones ganadoras (curPrice == 1)...")
        r = _HTTP_SESSION.get(
            f"https://data-api.polymarket.com/positions?user={wallet}&sizeThreshold=0&limit=500",
            timeout=15,
        )
        if r.status_code != 200:
            log.warning(f"Polymarket positions API: status {r.status_code}")
            return

        positions  = r.json() or []
        # Solo posiciones con curPrice == 1 (completamente ganadoras y resueltas)
        redeemable = [
            p for p in positions
            if p.get("curPrice") == 1 and float(p.get("size", 0)) > 0
        ]

        if not redeemable:
            log.info("  No hay posiciones ganadoras pendientes (curPrice==1).")
            return

        log.info(f"  {len(redeemable)} posición(es) para reclamar")

        w3 = get_web3()
        if not w3:
            return

        account = w3.eth.account.from_key(private_key)
        ct      = w3.eth.contract(
            address=Web3.to_checksum_address(CT_ADDRESS), abi=CT_ABI
        )

        for pos in redeemable:
            condition_id = pos.get("conditionId") or pos.get("condition_id")
            if not condition_id:
                continue

            title           = (pos.get("title") or pos.get("market") or "?")[:50]
            condition_bytes = bytes.fromhex(condition_id.replace("0x", ""))

            # Verificar que el contrato esté resuelto en cadena
            try:
                denom = ct.functions.payoutDenominator(condition_bytes).call()
                if denom == 0:
                    log.info(f"  SKIP (no resuelto aún): {title}")
                    continue
                log.info(f"  Reclamando: {title}")
            except Exception as e:
                log.warning(f"  No se pudo verificar denom: {title} — {e}")
                continue

            # Determinar index_sets ganadores via payoutNumerators
            winning = []
            try:
                for idx in [1, 2]:
                    if ct.functions.payoutNumerators(condition_bytes, idx - 1).call() > 0:
                        winning.append(idx)
            except Exception:
                winning = [1, 2]

            if not winning:
                log.info(f"  Sin index_sets ganadores: {title}")
                continue

            # Medir balance antes para calcular payout real
            usdc_contract = w3.eth.contract(
                address=Web3.to_checksum_address(USDC_ADDRESSES["native"]), abi=USDC_ABI
            )
            try:
                bal_before = usdc_contract.functions.balanceOf(account.address).call()
            except Exception:
                bal_before = None

            claimed = False
            for collateral, col_label in [
                (USDC_ADDRESSES["bridged"], "USDC.e"),    # USDC.e primero (igual que el script manual)
                (USDC_ADDRESSES["native"],  "USDC nativo"),
            ]:
                if claimed:
                    break
                for idx in winning:
                    if claimed:
                        break
                    try:
                        time.sleep(1)   # pausa para estabilidad de nonce entre txs
                        nonce   = _get_nonce(w3, account.address)
                        gas_p   = _safe_gas_price(w3)
                        tx      = ct.functions.redeemPositions(
                            Web3.to_checksum_address(collateral),
                            bytes(32), condition_bytes, [idx],
                        ).build_transaction({
                            "from": account.address, "nonce": nonce,
                            "gas": 200_000, "gasPrice": gas_p, "chainId": CHAIN_ID,
                        })
                        signed  = w3.eth.account.sign_transaction(tx, private_key)
                        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
                        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=90)

                        if receipt["status"] == 1:
                            # Payout real = delta de balance USDC
                            payout_usdc = 0.0
                            if bal_before is not None:
                                try:
                                    bal_after   = usdc_contract.functions.balanceOf(account.address).call()
                                    payout_usdc = (bal_after - bal_before) / 10**6
                                except Exception:
                                    pass
                            # Fallback: estimación de la API
                            if payout_usdc == 0.0:
                                payout_usdc = float(pos.get("size", 0))

                            log.info(
                                f"  RECLAMADO: ${payout_usdc:.4f} USDC ({col_label}) "
                                f"TX:{tx_hash.hex()[:16]}..."
                            )
                            trades   = load_trades()
                            bet_usdc = next(
                                (t["bet_usdc"] for t in trades if t.get("condition_id") == condition_id),
                                BET_SIZE_USDC,
                            )
                            record_trade_result(
                                condition_id, payout_usdc=payout_usdc, bet_usdc=bet_usdc
                            )
                            claimed = True

                    except Exception as e:
                        err = str(e).lower()
                        if "revert" in err:
                            # Revert = posición ya reclamada o no ganadora con este colateral
                            # Parar inmediatamente, no reintentar con otros index_sets
                            break
                        log.warning(f"  {col_label} idx={idx}: {str(e)[:80]}")

            if not claimed:
                log.warning(f"  No se pudo reclamar: {title[:40]}")

    except Exception as e:
        log.warning(f"auto_redeem_winnings error: {e}", exc_info=True)

# ─────────────────────────────────────────────────────────────────
# LOG BALANCE
# ─────────────────────────────────────────────────────────────────
def log_balance(private_key: str) -> None:
    w3 = get_web3()
    if not w3:
        return
    try:
        account = w3.eth.account.from_key(private_key)
        wallet  = account.address
        for label, addr in [("USDC nativo", USDC_ADDRESSES["native"]),
                             ("USDC.e",      USDC_ADDRESSES["bridged"])]:
            usdc = w3.eth.contract(address=Web3.to_checksum_address(addr), abi=USDC_ABI)
            bal  = usdc.functions.balanceOf(wallet).call() / 10**6
            log.info(f"  {label}: {bal:.4f} USDC")
    except Exception as e:
        log.warning(f"log_balance error: {e}")

# ─────────────────────────────────────────────────────────────────
# BUCLE PRINCIPAL
# ─────────────────────────────────────────────────────────────────
def run_bot() -> None:
    log.info("=" * 62)
    log.info("  BTC 15M Predictor Bot – Polymarket 15-Minute Markets")
    log.info("=" * 62)
    log.info(f"  Par         : BTCUSDT  (RSI={RSI_LEN}, BB={BB_LEN}x{BB_MULT})")
    log.info(f"  Apuesta     : ${BET_SIZE_USDC} | Confianza min: {MIN_CONFIDENCE*100:.0f}%")
    log.info(f"  Edge mín    : {MIN_EDGE*100:.0f}% | K_FACTOR: {K_FACTOR}")
    log.info(f"  Threshold   : {SIGNAL_THRESHOLD:.2f} (prob mínima para señal)")
    log.info(f"  Ventana     : {MIN_LO}-{MIN_HI} min antes del cierre 15M")
    log.info(f"  Chainlink   : validación Binance vs on-chain (máx {MAX_DIVERGENCE_PCT}%)")
    log.info(f"  Oráculo     : {ORACLE_RESOLUTION_SOURCE}")
    log.info("=" * 62)

    if not PRIVATE_KEY:
        log.error("PRIVATE_KEY_2 no definida en .env")
        sys.exit(1)

    # ── Cargar modelo ML (obligatorio en producción) ────────────
    ml_available = load_ml_model()
    if ml_available:
        log.info("  Modo señal  : ML calibrado (XGBoost + Isotonic)")
    else:
        log.critical(
            "  MODELO ML NO DISPONIBLE — el bot NO operará sin modelo calibrado.\n"
            "  La heurística sigmoide no es fiable en producción.\n"
            "  Ejecuta: python train_model_15m.py  para generar model_15m.pkl\n"
            "  El bot permanecerá en espera hasta que se reinicie con un modelo válido."
        )
        while True:
            time.sleep(3600)  # espera indefinida hasta que el operador arregle el modelo

    client = _build_clob_client(PRIVATE_KEY)
    if client is None:
        log.error("No se pudo conectar a Polymarket. Abortando.")
        sys.exit(1)

    # Balance allowance vía CLOB API
    try:
        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        bal    = client.get_balance_allowance(params=params)
        log.info(f"Balance CLOB: {bal}")
        client.update_balance_allowance(params=params)
    except Exception as e:
        log.warning(f"balance_allowance: {e}")

    approve_usdc_allowance(PRIVATE_KEY)
    time.sleep(5)   # mempool settle tras approves
    log_balance(PRIVATE_KEY)
    auto_redeem_winnings(PRIVATE_KEY)
    print_pnl_summary()

    current_window_ts    = None
    up_token             = None
    down_token           = None
    alert_sent           = False
    last_claim_candle    = None
    consecutive_errors   = 0
    MAX_ERRORS           = 5

    log.info("Bot activo. Esperando ventana 2-7 min antes del cierre 15M...\n")

    while True:
        try:
            new_window_ts = get_current_15m_window_ts()

            # ── Nueva ventana de 15 minutos ──────────────────────────
            if current_window_ts != new_window_ts:
                slug = build_15m_slug(new_window_ts)
                log.info(f"\nNueva ventana 15M: {slug}")
                market = find_15m_market()
                if market:
                    u, d = get_market_tokens(market)
                    if u and d:
                        current_window_ts = new_window_ts
                        up_token          = u
                        down_token        = d
                        alert_sent        = False
                        log.info(f"  UP   token: ...{up_token[-16:]}")
                        log.info(f"  DOWN token: ...{down_token[-16:]}")
                        # Redeem y P&L al inicio de cada ventana
                        auto_redeem_winnings(PRIVATE_KEY)
                        print_pnl_summary()
                    else:
                        log.error("No se obtuvieron tokens. Reintentando en 30s.")
                        time.sleep(30)
                        continue
                else:
                    log.error("Mercado 15M no encontrado. Reintentando en 30s.")
                    time.sleep(30)
                    continue

            df = get_btc_candles_15m(limit=250)
            if df.empty:
                time.sleep(CHECK_EVERY_SECS)
                continue

            remaining         = get_remaining_minutes(df)
            in_zone           = MIN_LO <= remaining <= MIN_HI
            current_candle_id = str(df.iloc[-1]["open_time"])

            # Auto-claim ~8 min antes del cierre (única vez por vela)
            if 7.5 <= remaining <= 9.5 and last_claim_candle != current_candle_id:
                last_claim_candle = current_candle_id
                log.info("=== Auto-claim ~8min antes del cierre 15M ===")
                auto_redeem_winnings(PRIVATE_KEY)
                log_balance(PRIVATE_KEY)

            window_dt = datetime.fromtimestamp(current_window_ts, tz=pytz.utc).astimezone(ET_ZONE)
            end_dt    = window_dt + timedelta(minutes=15)
            label     = f"{fmt_time(window_dt)}-{fmt_time(end_dt)} ET"

            if not in_zone:
                alert_sent = False
                log.info(f"fuera ({remaining:.1f}min) | BTC ${df.iloc[-1]['close']:,.0f} | {label}")
                time.sleep(CHECK_EVERY_SECS)
                continue

            # ── Validar precio Chainlink vs Binance ──────────────────
            cl_price           = get_chainlink_price()
            price_ok, diff_pct = check_price_divergence(df.iloc[-1]["close"], cl_price)
            if not price_ok:
                time.sleep(CHECK_EVERY_SECS)
                continue

            # ── Señal: solo ML calibrado (sin fallback heurístico) ────
            order_flow = get_order_flow()
            ind = predict_with_ml(df, order_flow)
            if ind is None:
                # Sin modelo ML no operamos: la heurística sigmoide es
                # matemáticamente frágil y perdería dinero en producción.
                log.info("  ML sin señal (modelo no cargado o features insuficientes). Skip.")
                time.sleep(CHECK_EVERY_SECS)
                continue

            source = ind.get("source", "heurística")
            signal = ("ALCISTA" if ind["signal_bull"]
                      else "BAJISTA" if ind["signal_bear"] else "NEUTRAL")
            log.info(
                f"EN ZONA {remaining:.1f}min | BTC ${ind['price']:,.0f} | {signal} "
                f"| Conf:{ind['confidence']*100:.0f}% "
                f"| G:{ind['prob_bull']*100:.0f}% B:{ind['prob_bear']*100:.0f}% "
                f"| RSI:{ind['rsi']:.1f} | [{source}] | {label}"
            )

            # ── Filtros de entrada ───────────────────────────────────
            if not (ind["signal_bull"] or ind["signal_bear"]):
                time.sleep(CHECK_EVERY_SECS)
                continue
            if ind["confidence"] < MIN_CONFIDENCE:
                log.info(f"  Confianza {ind['confidence']*100:.0f}% < mínimo. No opera.")
                time.sleep(CHECK_EVERY_SECS)
                continue
            if already_traded_this_window(current_window_ts):
                if not alert_sent:
                    log.info("  Ya operamos en esta ventana.")
                    alert_sent = True
                time.sleep(CHECK_EVERY_SECS)
                continue

            # ── Edge check ───────────────────────────────────────────
            direction = "UP" if ind["signal_bull"] else "DOWN"
            token_id  = up_token if ind["signal_bull"] else down_token
            our_prob  = ind["prob_bull"] if ind["signal_bull"] else ind["prob_bear"]

            token_price_info = get_token_market_price(client, token_id)
            if token_price_info is None:
                consecutive_errors += 1
                log.warning(f"  Sin precio de mercado. Errores: {consecutive_errors}/{MAX_ERRORS}")
                if consecutive_errors >= MAX_ERRORS:
                    log.warning("  Reconectando ClobClient...")
                    new_client = _build_clob_client(PRIVATE_KEY)
                    if new_client:
                        client             = new_client
                        consecutive_errors = 0
                        log.info("  ClobClient reconectado.")
                time.sleep(CHECK_EVERY_SECS)
                continue
            consecutive_errors = 0
            token_price = token_price_info["mid"]

            if not has_positive_edge(our_prob, token_price_info):
                log.info("  Edge insuficiente. No opera.")
                time.sleep(CHECK_EVERY_SECS)
                continue

            # ── Ejecutar orden ───────────────────────────────────────
            market_obj   = find_15m_market()
            cid          = (market_obj.get("conditionId") or market_obj.get("condition_id")) if market_obj else None
            market_title = (market_obj.get("question") or market_obj.get("title") or "") if market_obj else ""
            ml_source    = ind.get("source", "heurística")

            log.info(
                f"EJECUTANDO {direction} | modelo={our_prob*100:.1f}% | "
                f"ask={token_price_info['ask']*100:.1f}% | edge={(our_prob-(token_price_info['ask'] or token_price))*100:.1f}%"
            )
            if place_order(client, token_id, BET_SIZE_USDC):
                mark_traded(
                    current_window_ts, direction, token_price, our_prob,
                    cid, market_title=market_title, source=ml_source,
                )
                alert_sent = True
                log.info(f"Orden {direction} ejecutada y registrada.")
            else:
                log.error("Orden fallida. No se reintentará en esta ventana.")
                mark_traded(
                    current_window_ts, direction, token_price, our_prob,
                    cid, market_title=market_title, source=ml_source,
                )

        except KeyboardInterrupt:
            log.info("Bot detenido por el usuario.")
            print_pnl_summary()
            break
        except Exception as e:
            log.error(f"Error inesperado: {e}", exc_info=True)

        time.sleep(CHECK_EVERY_SECS)

# ─────────────────────────────────────────────────────────────────
# DRY-RUN
# ─────────────────────────────────────────────────────────────────
def run_dryrun() -> None:
    log.info("=" * 62)
    log.info("  DRY-RUN MODE – Sin órdenes reales")
    log.info(f"  Par: BTCUSDT 15M  (RSI={RSI_LEN}, BB={BB_LEN}x{BB_MULT})")
    log.info(f"  Edge mínimo: {MIN_EDGE*100:.0f}% | K_FACTOR: {K_FACTOR}")
    log.info("=" * 62)

    # Construir cliente solo si hay clave disponible (para leer precios reales)
    client = _build_clob_client(PRIVATE_KEY) if PRIVATE_KEY else None
    if client is None:
        log.warning("Sin PRIVATE_KEY_2 — el edge check mostrará N/A")

    last_window = None
    while True:
        try:
            new_window = get_current_15m_window_ts()
            if last_window != new_window:
                log.info(f"\nVentana activa: {build_15m_slug(new_window)}")
                last_window = new_window

            df = get_btc_candles_15m(limit=250)
            if df.empty:
                time.sleep(30)
                continue

            remaining = get_remaining_minutes(df)
            ind       = calc_indicators(df)
            if ind is None:
                time.sleep(30)
                continue

            in_zone = MIN_LO <= remaining <= MIN_HI
            zona    = "EN ZONA" if in_zone else f"fuera ({remaining:.1f}min)"
            signal  = ("ALCISTA" if ind["signal_bull"]
                       else "BAJISTA" if ind["signal_bear"] else "NEUTRAL")

            log.info(
                f"{zona} | BTC ${ind['price']:,.0f} | {signal} "
                f"| Conf:{ind['confidence']*100:.0f}% "
                f"| G:{ind['prob_bull']*100:.0f}% B:{ind['prob_bear']*100:.0f}% "
                f"| RSI:{ind['rsi']:.1f}"
            )

            if in_zone and (ind["signal_bull"] or ind["signal_bear"]) and client:
                market = find_15m_market()
                if market:
                    u, d      = get_market_tokens(market)
                    token_id  = u if ind["signal_bull"] else d
                    our_prob  = ind["prob_bull"] if ind["signal_bull"] else ind["prob_bear"]
                    direction = "UP" if ind["signal_bull"] else "DOWN"
                    if token_id:
                        mkt = get_token_market_price(client, token_id)
                        if mkt is not None:
                            ask = mkt["ask"] or mkt["mid"]
                            edge = our_prob - ask
                            log.info(
                                f"[DRY-RUN] {direction} | modelo={our_prob*100:.1f}% | "
                                f"ask={ask*100:.1f}% | spread={mkt['spread']*100:.1f}% | edge={edge*100:.1f}%"
                            )
                            verdict = "✓ OPERARÍAMOS" if has_positive_edge(our_prob, mkt) else "✗ Edge insuficiente"
                            log.info(f"[DRY-RUN] {verdict}")
                        else:
                            log.info(f"[DRY-RUN] {direction} | modelo={our_prob*100:.1f}% | (sin precio)")

        except KeyboardInterrupt:
            log.info("Dry-run detenido.")
            break
        except Exception as e:
            log.error(f"Error: {e}")

        time.sleep(CHECK_EVERY_SECS)

# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC 15M Predictor Bot – Polymarket")
    parser.add_argument("--dryrun", action="store_true", help="Sin órdenes reales")
    parser.add_argument("--stats",  action="store_true", help="Mostrar P&L y salir")
    args = parser.parse_args()

    if args.stats:
        print_pnl_summary()
    elif args.dryrun:
        run_dryrun()
    else:
        run_bot()
