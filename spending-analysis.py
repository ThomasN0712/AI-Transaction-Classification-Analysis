import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = "mastersheet_classified.csv"

MIN_CONFIDENCE = 0.60
EXCLUDE_CATEGORIES = {"Transfer"}  # remove if you want transfers included

def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"Date", "Amount", "Category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Found: {df.columns.tolist()}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Amount parsing (handles $, commas, parentheses)
    amt = (
        df["Amount"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    # Handle (123.45) as negative
    amt = amt.str.replace("(", "-", regex=False).str.replace(")", "", regex=False)

    df["Amount"] = pd.to_numeric(amt, errors="coerce")
    df = df.dropna(subset=["Amount"])

    # Confidence filter if present
    if "Confidence" in df.columns:
        df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
        df = df[df["Confidence"].fillna(0) >= MIN_CONFIDENCE]

    df["Category"] = df["Category"].astype(str).str.strip()
    df.loc[df["Category"].eq("") | df["Category"].eq("nan"), "Category"] = "Misc"

    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    # Since your file is spending-only, treat magnitude as spend
    df["Spend"] = df["Amount"].abs()

    # Optional: if you truly have no income, you can also force Spend = Amount (if all positive)
    # df["Spend"] = df["Amount"]

    return df

def quick_stats(df: pd.DataFrame) -> None:
    total_spend = df["Spend"].sum()
    months = df["Month"].nunique()
    rows = len(df)

    print("Summary")
    print(f"Rows analyzed: {rows}")
    print(f"Months: {months}")
    print(f"Total spending: {total_spend:,.2f}")
    if months > 0:
        print(f"Avg monthly spending: {(total_spend / months):,.2f}")

    top = (
        df.groupby("Category")["Spend"]
        .sum()
        .sort_values(ascending=False)
    )
    print("\nTop spending categories")
    print(top.head(12).to_string())

def chart_spend_by_category(df: pd.DataFrame) -> None:
    spend = (
        df.groupby("Category")["Spend"]
        .sum()
        .sort_values(ascending=False)
    )
    spend = spend[spend > 0]

    if spend.empty:
        print("No spend data to plot in chart_spend_by_category.")
        return

    plt.figure()
    spend.head(12).plot(kind="bar")
    plt.title("Top Categories by Total Spending")
    plt.xlabel("Category")
    plt.ylabel("Spending")
    plt.tight_layout()
    plt.savefig("chart_spend_by_category.png", dpi=160)
    plt.close()

def chart_monthly_spend(df: pd.DataFrame) -> None:
    monthly = df.groupby("Month")["Spend"].sum().sort_index()

    if monthly.empty:
        print("No spend data to plot in chart_monthly_spend.")
        return

    plt.figure()
    monthly.plot(kind="line")
    plt.title("Monthly Spending Trend")
    plt.xlabel("Month")
    plt.ylabel("Spending")
    plt.tight_layout()
    plt.savefig("chart_monthly_spend.png", dpi=160)
    plt.close()

def chart_category_share_pie(df: pd.DataFrame) -> None:
    spend = (
        df.groupby("Category")["Spend"]
        .sum()
        .sort_values(ascending=False)
    )
    spend = spend[spend > 0]

    if spend.empty:
        print("No spend data to plot in chart_category_share_pie.")
        return

    top_n = 8
    top = spend.head(top_n)
    other = spend.iloc[top_n:].sum()
    if other > 0:
        top = pd.concat([top, pd.Series({"Other": other})])

    plt.figure()
    top.plot(kind="pie", autopct="%1.1f%%")
    plt.title("Spending Share by Category")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("chart_category_share.png", dpi=160)
    plt.close()

def month_over_month(df: pd.DataFrame) -> None:
    monthly = df.groupby("Month")["Spend"].sum().sort_index()
    if len(monthly) < 2:
        return

    last = monthly.iloc[-1]
    prev = monthly.iloc[-2]
    delta = last - prev
    pct = (delta / prev * 100) if prev != 0 else None

    print("\nMonth-over-month")
    if pct is None:
        print(f"Last month spending change: {delta:,.2f}")
    else:
        print(f"Last month spending change: {delta:,.2f} ({pct:.1f}%)")

def largest_swings(df: pd.DataFrame) -> None:
    # Compare last month vs average of prior months for each category
    if df["Month"].nunique() < 3:
        return

    last_month = df["Month"].max()
    prior = df[df["Month"] < last_month]
    last_df = df[df["Month"] == last_month]

    prior_avg = prior.groupby("Category")["Spend"].sum() / prior["Month"].nunique()
    last_sum = last_df.groupby("Category")["Spend"].sum()

    swing = (last_sum - prior_avg).sort_values(ascending=False)

    print("\nLargest category overspends vs prior-month average (top 10)")
    print(swing.head(10).to_string())

def main():
    df = load_df(INPUT_CSV)

    if EXCLUDE_CATEGORIES:
        df = df[~df["Category"].isin(EXCLUDE_CATEGORIES)].copy()

    quick_stats(df)
    month_over_month(df)
    largest_swings(df)

    chart_spend_by_category(df)
    chart_monthly_spend(df)
    chart_category_share_pie(df)

    print("\nSaved charts:")
    print("chart_spend_by_category.png")
    print("chart_monthly_spend.png")
    print("chart_category_share.png")

if __name__ == "__main__":
    main()
