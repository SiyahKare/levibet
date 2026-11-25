"""Plain-text signal messages (Turkish / English). No guarantees, no stake advice."""

from app.core.config import settings
from app.db.models import Match, Signal

DISCLAIMER = {
    "tr": (
        "Bu mesaj istatistiksel model çıktısıdır; bahis tavsiyesi değildir, kazanç vaadi içermez. "
        "Geçmiş performans geleceği garanti etmez. 18 yaş ve üzeri içindir; yerel mevzuata uyun."
    ),
    "en": (
        "Statistical model output, not betting advice and no promise of profit. "
        "Past performance does not guarantee future results. 18+ only; follow your local laws."
    ),
}

SELECTION_NAME = {
    "tr": {"H": "Ev sahibi galibiyeti", "D": "Beraberlik", "A": "Deplasman galibiyeti"},
    "en": {"H": "Home win", "D": "Draw", "A": "Away win"},
}

LABELS = {
    "tr": ("Maç", "Lig", "Başlangıç (UTC)", "Seçim", "Oran", "Model olasılığı",
           "Piyasa olasılığı", "Tahmini avantaj", "Temkinli avantaj"),
    "en": ("Match", "League", "Kickoff (UTC)", "Pick", "Odds", "Model probability",
           "Market probability", "Estimated edge", "Conservative edge"),
}


def format_signal(sig: Signal, match: Match, lang: str | None = None) -> str:
    lang = lang if lang in DISCLAIMER else settings.SIGNAL_LANG
    lang = lang if lang in DISCLAIMER else "en"
    m, lg, ko, pk, od, mp, kp, ed, ce = LABELS[lang]
    return "\n".join(
        [
            f"{m}: {match.home_team} - {match.away_team}",
            f"{lg}: {match.league}",
            f"{ko}: {match.kickoff_at:%Y-%m-%d %H:%M}",
            f"{pk}: {SELECTION_NAME[lang][sig.selection]}",
            f"{od}: {sig.odds:.2f} ({sig.bookmaker})",
            f"{mp}: {sig.model_prob * 100:.1f}% (±{sig.prob_std * 100:.1f})",
            f"{kp}: {sig.market_prob * 100:.1f}%",
            f"{ed}: {sig.edge * 100:+.1f}%",
            f"{ce}: {sig.conservative_edge * 100:+.1f}%",
            "",
            DISCLAIMER[lang],
        ]
    )
