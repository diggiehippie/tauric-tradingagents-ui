import streamlit as st
from datetime import date, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

# Inject Streamlit secrets into environment (for Streamlit Cloud)
for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY", "OPENROUTER_API_KEY"]:
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.propagation import Propagator

st.set_page_config(
    page_title="TradingAgents",
    page_icon="📈",
    layout="wide",
)

st.title("📈 TradingAgents")
st.caption("Multi-Agent LLM Financial Trading Framework by [Tauric Research](https://github.com/TauricResearch/TradingAgents)")

# ── Sidebar: configuration ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    llm_provider = st.selectbox(
        "LLM Provider",
        ["openai", "anthropic", "google"],
        index=0,
    )

    provider_models = {
        "openai": ["gpt-5.4-mini", "gpt-5.4", "gpt-4o", "gpt-4o-mini"],
        "anthropic": ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001"],
        "google": ["gemini-2.0-flash", "gemini-2.5-pro"],
    }

    deep_model = st.selectbox(
        "Deep Think Model",
        provider_models[llm_provider],
        index=0,
    )
    quick_model = st.selectbox(
        "Quick Think Model",
        provider_models[llm_provider],
        index=0,
    )

    st.divider()
    st.subheader("Analysts")
    use_market = st.checkbox("Market Analyst", value=True)
    use_social = st.checkbox("Social Analyst", value=True)
    use_news = st.checkbox("News Analyst", value=True)
    use_fundamentals = st.checkbox("Fundamentals Analyst", value=True)

    st.divider()
    st.subheader("Debate Settings")
    max_debate_rounds = st.slider("Debate Rounds", 1, 3, 1)
    max_risk_rounds = st.slider("Risk Discussion Rounds", 1, 3, 1)

# ── Main: input form ──────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])
with col1:
    ticker = st.text_input("Stock Ticker", value="NVDA", placeholder="e.g. AAPL, MSFT, NVDA").upper()
with col2:
    trade_date = st.date_input(
        "Analysis Date",
        value=date.today() - timedelta(days=1),
        max_value=date.today() - timedelta(days=1),
    )

run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ── Analysis runner ───────────────────────────────────────────────────────────
if run_button:
    selected_analysts = []
    if use_market:
        selected_analysts.append("market")
    if use_social:
        selected_analysts.append("social")
    if use_news:
        selected_analysts.append("news")
    if use_fundamentals:
        selected_analysts.append("fundamentals")

    if not selected_analysts:
        st.error("Select at least one analyst.")
        st.stop()

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = llm_provider
    config["deep_think_llm"] = deep_model
    config["quick_think_llm"] = quick_model
    config["max_debate_rounds"] = max_debate_rounds
    config["max_risk_discuss_rounds"] = max_risk_rounds

    st.divider()
    st.subheader(f"Analysis: {ticker} — {trade_date}")

    # Placeholders for live updates
    status_placeholder = st.empty()

    # Section containers (shown as they complete)
    section_titles = {
        "market_report": "📊 Market Analysis",
        "sentiment_report": "💬 Social Sentiment",
        "news_report": "📰 News Analysis",
        "fundamentals_report": "🏦 Fundamentals Analysis",
        "investment_plan": "🤝 Research Team Decision",
        "trader_investment_plan": "💼 Trading Team Plan",
        "final_trade_decision": "🎯 Final Trade Decision",
    }

    # Map state keys to relevant analysts (None = always shown)
    section_analyst_map = {
        "market_report": "market",
        "sentiment_report": "social",
        "news_report": "news",
        "fundamentals_report": "fundamentals",
        "investment_plan": None,
        "trader_investment_plan": None,
        "final_trade_decision": None,
    }

    active_sections = {
        k for k, v in section_analyst_map.items()
        if v is None or v in selected_analysts
    }

    # Pre-create expanders for each section
    section_containers = {}
    for key in section_titles:
        if key in active_sections:
            section_containers[key] = st.expander(section_titles[key], expanded=False)

    # Final decision gets its own prominent spot
    decision_placeholder = st.empty()

    try:
        with st.spinner(f"Initializing agents for {ticker}..."):
            ta = TradingAgentsGraph(
                selected_analysts=selected_analysts,
                debug=False,
                config=config,
            )
            propagator = Propagator(max_recur_limit=config["max_recur_limit"])
            init_state = propagator.create_initial_state(ticker, str(trade_date))
            graph_args = propagator.get_graph_args()

        displayed_sections = set()
        current_state = {}

        agent_sequence = []
        if "market" in selected_analysts:
            agent_sequence.append("Market Analyst")
        if "social" in selected_analysts:
            agent_sequence.append("Social Analyst")
        if "news" in selected_analysts:
            agent_sequence.append("News Analyst")
        if "fundamentals" in selected_analysts:
            agent_sequence.append("Fundamentals Analyst")
        agent_sequence += [
            "Bull Researcher", "Bear Researcher", "Research Manager",
            "Trader",
            "Aggressive Analyst", "Neutral Analyst", "Conservative Analyst",
            "Portfolio Manager",
        ]
        total_agents = len(agent_sequence)
        completed_agents = 0

        for chunk in ta.graph.stream(init_state, **graph_args):
            current_state.update(chunk)

            # Detect which agent just ran by looking at new non-empty sections
            for section_key in active_sections:
                if section_key in displayed_sections:
                    continue
                content = current_state.get(section_key, "")
                if not content:
                    continue

                # Show this section
                displayed_sections.add(section_key)
                completed_agents = min(completed_agents + 1, total_agents)

                with section_containers[section_key]:
                    st.markdown(content)

            # Update status bar
            progress = completed_agents / total_agents
            status_placeholder.progress(
                progress,
                text=f"Running analysis... ({completed_agents}/{total_agents} sections complete)",
            )

        # Final decision banner
        final_decision = current_state.get("final_trade_decision", "")
        processed = ta.process_signal(final_decision)

        status_placeholder.success(f"Analysis complete for **{ticker}** on {trade_date}")

        # Expand the final decision section
        if "final_trade_decision" in section_containers:
            section_containers["final_trade_decision"].expanded = True

        # Big decision banner
        color_map = {"BUY": "green", "SELL": "red", "HOLD": "orange"}
        color = color_map.get(processed.upper(), "blue")

        decision_placeholder.markdown(
            f"""
            <div style="
                background-color: {'#d4edda' if processed.upper() == 'BUY' else '#f8d7da' if processed.upper() == 'SELL' else '#fff3cd'};
                border: 2px solid {'#28a745' if processed.upper() == 'BUY' else '#dc3545' if processed.upper() == 'SELL' else '#ffc107'};
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                margin-top: 16px;
            ">
                <h2 style="margin: 0; color: {'#155724' if processed.upper() == 'BUY' else '#721c24' if processed.upper() == 'SELL' else '#856404'};">
                    Final Decision: {processed.upper()}
                </h2>
                <p style="margin: 4px 0 0 0; color: #666;">
                    {ticker} · {trade_date}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Error during analysis: {e}")
        raise
