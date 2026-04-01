from src.agent_tools import run_tool

tools_to_test = [
    "fetch_market_data",
    "preprocess_market_data",
    "run_feature_engineering",
    "check_news_sentiment",
    "run_decisions",
    "apply_risk_controls",
    "run_backtest",
]

for tool_name in tools_to_test:
    print(f"\n=== Testing {tool_name} ===")
    result = run_tool(tool_name)
    print(result["status"])
    print(result["summary"])