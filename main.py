from src.config import CRYPTO_ASSETS, START_DATE, END_DATE


def main() -> None:
    print("Agentic AI Trading Workflow Project")
    print(f"Assets selected: {len(CRYPTO_ASSETS)}")
    print(f"Date range: {START_DATE} to {END_DATE}")


if __name__ == "__main__":
    main()