from pathlib import Path
import json
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Utility helpers
# ============================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names so plot_id / Plot_ID / plot-id become Plot Id.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("_", " ", regex=False)
        .str.replace("-", " ", regex=False)
        .str.title()
    )
    return df


def read_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(path)
    df = clean_column_names(df)

    print(f"\nLoaded {filename}")
    print("Columns found:")
    print(list(df.columns))

    return df


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return 0.0
    return float(numerator / denominator)


def round_number(value, decimals=3):
    if pd.isna(value):
        return None
    return round(float(value), decimals)


def write_json(filename: str, data: dict):
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    print(f"Created {output_path}")


def add_delay_bucket(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates Delay Bucket if it is not already in the CSV.
    """
    df = sensor_df.copy()

    required_columns = ["Alert Flag", "Action Taken", "Action Delay Days"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Cannot create Delay Bucket. Missing column: {column}")

    def classify_delay(row):
        if row["Alert Flag"] != 1:
            return "No alert"

        if row["Action Taken"] != 1:
            return "No response"

        if row["Action Delay Days"] == 0:
            return "Same-day"

        if row["Action Delay Days"] == 1:
            return "1-day delay"

        return "2+ day delay"

    if "Delay Bucket" not in df.columns:
        df["Delay Bucket"] = df.apply(classify_delay, axis=1)

    return df


# ============================================================
# Advisor Tool 1: Plot Priority
# ============================================================

def rank_priority_plots(sensor_df: pd.DataFrame) -> dict:
    required_columns = [
        "Plot Id",
        "Alert Flag",
        "Plant Stress Index",
        "Action Delay Days",
        "Action Taken",
    ]

    for column in required_columns:
        if column not in sensor_df.columns:
            raise ValueError(f"Missing column in daily_sensor_readings.csv: {column}")

    df = sensor_df.copy()

    df["no_response_case"] = (
        (df["Alert Flag"] == 1) & (df["Action Taken"] == 0)
    ).astype(int)

    grouped = (
        df.groupby("Plot Id")
        .agg(
            total_alerts=("Alert Flag", "sum"),
            avg_plant_stress=("Plant Stress Index", "mean"),
            avg_response_delay=("Action Delay Days", "mean"),
            no_response_count=("no_response_case", "sum"),
        )
        .reset_index()
    )

    grouped["urgency_score"] = (
        grouped["avg_plant_stress"]
        + grouped["total_alerts"]
        + grouped["avg_response_delay"]
        + grouped["no_response_count"]
    )

    grouped = grouped.sort_values("urgency_score", ascending=False)

    top_plots = []
    for _, row in grouped.head(10).iterrows():
        top_plots.append(
            {
                "plot_id": row["Plot Id"],
                "urgency_score": round_number(row["urgency_score"], 3),
                "total_alerts": int(row["total_alerts"]),
                "avg_plant_stress": round_number(row["avg_plant_stress"], 3),
                "avg_response_delay_days": round_number(row["avg_response_delay"], 3),
                "no_response_count": int(row["no_response_count"]),
                "reason": (
                    "High urgency score based on alert count, plant stress, "
                    "response delay, and unresolved alert cases."
                ),
            }
        )

    return {
        "summary_type": "plot_priority_summary",
        "method": (
            "Ranks plots using alert count, average plant stress, "
            "average response delay, and no-response cases."
        ),
        "top_priority_plots": top_plots,
    }


# ============================================================
# Advisor Tool 2: Response Timing
# ============================================================

def compare_response_timing(sensor_df: pd.DataFrame) -> dict:
    required_columns = [
        "Delay Bucket",
        "Alert Type",
        "Alert Flag",
        "Action Taken",
        "Action Delay Days",
        "Post Action Stress Delta 3D",
    ]

    for column in required_columns:
        if column not in sensor_df.columns:
            raise ValueError(f"Missing column in daily_sensor_readings.csv: {column}")

    alert_df = sensor_df[sensor_df["Alert Flag"] == 1].copy()

    alert_df["same_day_response"] = (
        (alert_df["Action Taken"] == 1)
        & (alert_df["Action Delay Days"] == 0)
    ).astype(int)

    alert_df["worsening_stress_case"] = (
        alert_df["Post Action Stress Delta 3D"] > 0
    ).astype(int)

    delay_summary_df = (
        alert_df.groupby("Delay Bucket")
        .agg(
            alert_count=("Alert Flag", "sum"),
            avg_stress_recovery=("Post Action Stress Delta 3D", "mean"),
            avg_response_delay=("Action Delay Days", "mean"),
            same_day_response_count=("same_day_response", "sum"),
            worsening_stress_count=("worsening_stress_case", "sum"),
        )
        .reset_index()
    )

    delay_summary = []
    for _, row in delay_summary_df.iterrows():
        delay_summary.append(
            {
                "delay_bucket": row["Delay Bucket"],
                "alert_count": int(row["alert_count"]),
                "avg_stress_recovery": round_number(row["avg_stress_recovery"], 3),
                "avg_response_delay_days": round_number(row["avg_response_delay"], 3),
                "same_day_response_rate": round_number(
                    safe_divide(row["same_day_response_count"], row["alert_count"]),
                    3,
                ),
                "worsening_stress_count": int(row["worsening_stress_count"]),
            }
        )

    alert_type_df = (
        alert_df.groupby("Alert Type")
        .agg(
            alert_count=("Alert Flag", "sum"),
            avg_stress_recovery=("Post Action Stress Delta 3D", "mean"),
            avg_response_delay=("Action Delay Days", "mean"),
            worsening_stress_count=("worsening_stress_case", "sum"),
        )
        .reset_index()
        .sort_values("avg_response_delay", ascending=False)
    )

    alert_type_summary = []
    for _, row in alert_type_df.iterrows():
        alert_type_summary.append(
            {
                "alert_type": row["Alert Type"],
                "alert_count": int(row["alert_count"]),
                "avg_stress_recovery": round_number(row["avg_stress_recovery"], 3),
                "avg_response_delay_days": round_number(row["avg_response_delay"], 3),
                "worsening_stress_count": int(row["worsening_stress_count"]),
            }
        )

    return {
        "summary_type": "response_timing_summary",
        "important_note": (
            "Negative stress delta means plant stress improved. Positive stress delta "
            "means stress worsened. These results show association, not proven causality."
        ),
        "delay_bucket_summary": delay_summary,
        "alert_type_summary": alert_type_summary,
    }


# ============================================================
# Advisor Tool 3: Precision ROI
# ============================================================

def summarize_precision_roi(
    input_costs_df: pd.DataFrame,
    input_applications_df: pd.DataFrame,
    season_df: pd.DataFrame,
) -> dict:
    required_input_cost_columns = [
        "Daily Precision Cost",
        "Daily Routine Cost",
        "Daily Precision Actions Count",
        "Daily Total Actions Count",
        "Daily Total Input Cost",
    ]

    required_season_columns = [
        "Precision Benefit Cad",
        "Season Roi",
        "Season Profit Cad",
        "Season Yield Kg M2",
        "Marketable Ratio",
    ]

    for column in required_input_cost_columns:
        if column not in input_costs_df.columns:
            raise ValueError(f"Missing column in daily_input_costs.csv: {column}")

    for column in required_season_columns:
        if column not in season_df.columns:
            raise ValueError(f"Missing column in season_summary.csv: {column}")

    precision_cost = input_costs_df["Daily Precision Cost"].sum()
    routine_cost = input_costs_df["Daily Routine Cost"].sum()
    total_input_cost = input_costs_df["Daily Total Input Cost"].sum()

    precision_action_count = input_costs_df["Daily Precision Actions Count"].sum()
    routine_action_count = (
        input_costs_df["Daily Total Actions Count"].sum() - precision_action_count
    )

    precision_cost_share = safe_divide(precision_cost, total_input_cost)

    input_type_summary = []
    if (
        not input_applications_df.empty
        and "Input Type" in input_applications_df.columns
        and "Total Cost" in input_applications_df.columns
    ):
        input_grouped = (
            input_applications_df.groupby("Input Type")
            .agg(total_cost=("Total Cost", "sum"))
            .reset_index()
            .sort_values("total_cost", ascending=False)
        )

        for _, row in input_grouped.iterrows():
            input_type_summary.append(
                {
                    "input_type": row["Input Type"],
                    "total_cost": round_number(row["total_cost"], 2),
                }
            )

    return {
        "summary_type": "precision_roi_summary",
        "precision_cost": round_number(precision_cost, 2),
        "routine_cost": round_number(routine_cost, 2),
        "precision_action_count": int(precision_action_count),
        "routine_action_count": int(routine_action_count),
        "precision_cost_share": round_number(precision_cost_share, 4),
        "total_precision_benefit": round_number(
            season_df["Precision Benefit Cad"].sum(), 2
        ),
        "avg_season_roi": round_number(season_df["Season Roi"].mean(), 4),
        "total_season_profit": round_number(season_df["Season Profit Cad"].sum(), 2),
        "avg_yield_kg_m2": round_number(season_df["Season Yield Kg M2"].mean(), 3),
        "avg_marketable_ratio": round_number(season_df["Marketable Ratio"].mean(), 4),
        "input_type_cost_summary": input_type_summary,
    }


# ============================================================
# Advisor Tool 4: Financing Memo Summary
# ============================================================

def generate_financing_memo_summary(season_df: pd.DataFrame) -> dict:
    required_columns = [
        "Plot Id",
        "Season Roi",
        "Season Profit Cad",
        "Season Revenue Cad",
        "Total Cost Cad",
        "Precision Benefit Cad",
        "Season Yield Kg M2",
        "Marketable Ratio",
    ]

    for column in required_columns:
        if column not in season_df.columns:
            raise ValueError(f"Missing column in season_summary.csv: {column}")

    top_roi_df = season_df.sort_values("Season Roi", ascending=False).head(5)
    top_benefit_df = season_df.sort_values("Precision Benefit Cad", ascending=False).head(5)

    top_roi_plots = []
    for _, row in top_roi_df.iterrows():
        top_roi_plots.append(
            {
                "plot_id": row["Plot Id"],
                "season_roi": round_number(row["Season Roi"], 4),
                "season_profit_cad": round_number(row["Season Profit Cad"], 2),
                "precision_benefit_cad": round_number(row["Precision Benefit Cad"], 2),
            }
        )

    top_precision_benefit_plots = []
    for _, row in top_benefit_df.iterrows():
        top_precision_benefit_plots.append(
            {
                "plot_id": row["Plot Id"],
                "precision_benefit_cad": round_number(row["Precision Benefit Cad"], 2),
                "season_roi": round_number(row["Season Roi"], 4),
                "season_profit_cad": round_number(row["Season Profit Cad"], 2),
            }
        )

    return {
        "summary_type": "financing_memo_summary",
        "avg_season_roi": round_number(season_df["Season Roi"].mean(), 4),
        "total_profit_cad": round_number(season_df["Season Profit Cad"].sum(), 2),
        "total_revenue_cad": round_number(season_df["Season Revenue Cad"].sum(), 2),
        "total_cost_cad": round_number(season_df["Total Cost Cad"].sum(), 2),
        "total_precision_benefit_cad": round_number(
            season_df["Precision Benefit Cad"].sum(), 2
        ),
        "avg_yield_kg_m2": round_number(season_df["Season Yield Kg M2"].mean(), 3),
        "avg_marketable_ratio": round_number(season_df["Marketable Ratio"].mean(), 4),
        "top_roi_plots": top_roi_plots,
        "top_precision_benefit_plots": top_precision_benefit_plots,
    }


# ============================================================
# Markdown output
# ============================================================

def create_advisor_recommendations_md(
    plot_priority: dict,
    precision_roi: dict,
    financing_memo: dict,
):
    top_plot = plot_priority["top_priority_plots"][0]
    precision_cost_share_pct = precision_roi["precision_cost_share"] * 100
    avg_roi_pct = financing_memo["avg_season_roi"] * 100
    marketable_pct = financing_memo["avg_marketable_ratio"] * 100

    markdown = f"""# GreenLeaf Advisor Recommendations

## 1. Plot Priority

The highest-priority plot is **{top_plot["plot_id"]}** with an urgency score of **{top_plot["urgency_score"]}**.

This plot should be reviewed first because its score combines:
- total alerts
- average plant stress
- response delay
- unresolved alert cases

## 2. Response Timing

Response timing results should be interpreted as **association, not proven causality**.

Negative stress delta means plant stress improved. Positive stress delta means plant stress worsened.

The advisor should prioritize alert types and delay buckets that show slower response timing or weaker stress recovery.

## 3. Precision ROI

Precision actions accounted for **{precision_cost_share_pct:.1f}%** of total input cost.

Total precision benefit was **${precision_roi["total_precision_benefit"]:,.0f}**.

This suggests that precision-supported decisions can be discussed as a business value driver when paired with verified cost, ROI, and season outcome data.

## 4. Financing Summary

Across season outcomes:
- Average season ROI: **{avg_roi_pct:.1f}%**
- Total season profit: **${financing_memo["total_profit_cad"]:,.0f}**
- Total precision benefit: **${financing_memo["total_precision_benefit_cad"]:,.0f}**
- Average yield: **{financing_memo["avg_yield_kg_m2"]:.1f} kg/m2**
- Average marketable ratio: **{marketable_pct:.1f}%**

These verified metrics can support a lender-ready financing memo because they connect cost, profit, ROI, yield, marketability, and precision benefit.

## Advisor Guardrails

The advisor must:
- use only verified Tableau/Python-generated KPI values
- avoid inventing unsupported numbers
- describe response timing as association, not proven causality
- return structured JSON before plain-English recommendations
- explain every recommendation using available KPI evidence
"""

    output_path = OUTPUT_DIR / "advisor_recommendations.md"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(markdown)

    print(f"Created {output_path}")


# ============================================================
# Tableau-ready CSV outputs for Dashboard 5
# ============================================================

def write_tableau_ready_csvs(
    plot_priority: dict,
    precision_roi: dict,
    financing_memo: dict,
):
    top_plot = plot_priority["top_priority_plots"][0]
    precision_cost_share_pct = precision_roi["precision_cost_share"] * 100
    avg_roi_pct = financing_memo["avg_season_roi"] * 100
    marketable_pct = financing_memo["avg_marketable_ratio"] * 100

    # D5 KPI cards
    advisor_kpi_cards = pd.DataFrame(
        [
            {
                "Card Order": 1,
                "Metric": "Advisor Tools",
                "Value": "4",
                "Label": "Advisor Tools",
                "Category": "System",
            },
            {
                "Card Order": 2,
                "Metric": "KPI Groups",
                "Value": "5",
                "Label": "KPI Groups",
                "Category": "System",
            },
            {
                "Card Order": 3,
                "Metric": "Structured Outputs",
                "Value": "JSON + CSV",
                "Label": "Structured Outputs",
                "Category": "Output",
            },
            {
                "Card Order": 4,
                "Metric": "Guardrail Rules",
                "Value": "No Hallucination",
                "Label": "Guardrail Rules",
                "Category": "Trust",
            },
            {
                "Card Order": 5,
                "Metric": "Output Types",
                "Value": "Farmer + Lender",
                "Label": "Output Types",
                "Category": "Output",
            },
        ]
    )
    advisor_kpi_cards.to_csv(OUTPUT_DIR / "advisor_kpi_cards.csv", index=False)

    # D5 tool outputs
    advisor_tool_outputs = pd.DataFrame(
        [
            {
                "Tool Order": 1,
                "Tool Name": "rank_priority_plots",
                "Purpose": "Ranks high-urgency plots using alert count, plant stress, response delay, and no-response cases.",
                "Verified Inputs": "total_alerts, avg_plant_stress, avg_response_delay_days, no_response_count, urgency_score",
                "Output": "Plot priority list plus reason.",
            },
            {
                "Tool Order": 2,
                "Tool Name": "compare_response_timing",
                "Purpose": "Compares response speed with post-action stress recovery by delay bucket and alert type.",
                "Verified Inputs": "delay_bucket, alert_type, avg_stress_recovery, avg_response_delay_days, same_day_response_rate",
                "Output": "Timing bottleneck summary.",
            },
            {
                "Tool Order": 3,
                "Tool Name": "summarize_precision_roi",
                "Purpose": "Connects precision cost, routine cost, action count, precision benefit, and season ROI.",
                "Verified Inputs": "precision_cost, routine_cost, precision_action_count, precision_cost_share, avg_season_roi",
                "Output": "ROI and cost-effectiveness summary.",
            },
            {
                "Tool Order": 4,
                "Tool Name": "generate_financing_memo",
                "Purpose": "Turns verified profit, cost, ROI, yield, marketability, and precision benefit into a lender-ready summary.",
                "Verified Inputs": "total_profit_cad, total_cost_cad, avg_season_roi, avg_yield_kg_m2, avg_marketable_ratio",
                "Output": "Financing memo draft.",
            },
        ]
    )
    advisor_tool_outputs.to_csv(OUTPUT_DIR / "advisor_tool_outputs.csv", index=False)

    # D5 advisor recommendations
    advisor_recommendations_table = pd.DataFrame(
        [
            {
                "Recommendation Order": 1,
                "Area": "Plot Priority",
                "Finding": (
                    f"Top priority plot is {top_plot['plot_id']} "
                    f"with urgency score {top_plot['urgency_score']}."
                ),
                "Recommendation": (
                    "Review the highest-urgency plots first because they combine "
                    "alert volume, plant stress, response delay, and unresolved alert cases."
                ),
                "Evidence": (
                    f"{top_plot['plot_id']} has {top_plot['total_alerts']} alerts, "
                    f"avg stress {top_plot['avg_plant_stress']}, "
                    f"avg delay {top_plot['avg_response_delay_days']} days, "
                    f"and {top_plot['no_response_count']} no-response cases."
                ),
            },
            {
                "Recommendation Order": 2,
                "Area": "Response Timing",
                "Finding": "Response timing patterns are linked with stress recovery outcomes.",
                "Recommendation": (
                    "Prioritize same-day response workflows for high-risk alerts and "
                    "review slower alert categories for bottlenecks."
                ),
                "Evidence": "Response timing results are association only, not proven causality.",
            },
            {
                "Recommendation Order": 3,
                "Area": "Precision ROI",
                "Finding": (
                    f"Precision actions account for {precision_cost_share_pct:.1f}% "
                    "of total input cost."
                ),
                "Recommendation": (
                    "Use precision actions as targeted interventions where cost, ROI, "
                    "and season outcome evidence support them."
                ),
                "Evidence": (
                    f"Precision cost was ${precision_roi['precision_cost']:,.0f}, "
                    f"routine cost was ${precision_roi['routine_cost']:,.0f}, "
                    f"and total precision benefit was ${precision_roi['total_precision_benefit']:,.0f}."
                ),
            },
            {
                "Recommendation Order": 4,
                "Area": "Financing Summary",
                "Finding": (
                    f"Average season ROI is {avg_roi_pct:.1f}% with total season "
                    f"profit of ${financing_memo['total_profit_cad']:,.0f}."
                ),
                "Recommendation": (
                    "Use ROI, profit, yield, marketability, and precision benefit metrics "
                    "to support lender-ready investment discussion."
                ),
                "Evidence": (
                    f"Total precision benefit was ${financing_memo['total_precision_benefit_cad']:,.0f}, "
                    f"avg yield was {financing_memo['avg_yield_kg_m2']:.1f} kg/m2, "
                    f"and avg marketable ratio was {marketable_pct:.1f}%."
                ),
            },
        ]
    )
    advisor_recommendations_table.to_csv(
        OUTPUT_DIR / "advisor_recommendations_table.csv", index=False
    )

    # D5 guardrails
    advisor_guardrails = pd.DataFrame(
        [
            {
                "Rule Order": 1,
                "Guardrail": "Use verified KPI outputs only",
                "Reason": "The advisor should use Tableau/Python-generated metrics, not raw guesses.",
            },
            {
                "Rule Order": 2,
                "Guardrail": "Do not invent numbers",
                "Reason": "Recommendations must not contain unsupported costs, ROI values, yields, or plot claims.",
            },
            {
                "Rule Order": 3,
                "Guardrail": "Return structured JSON first",
                "Reason": "Structured outputs make the advisor easier to validate and reuse.",
            },
            {
                "Rule Order": 4,
                "Guardrail": "Association, not causality",
                "Reason": "Response timing patterns do not prove causation by themselves.",
            },
            {
                "Rule Order": 5,
                "Guardrail": "Explain recommendation evidence",
                "Reason": "Every recommendation should be connected to available KPI evidence.",
            },
        ]
    )
    advisor_guardrails.to_csv(OUTPUT_DIR / "advisor_guardrails.csv", index=False)

    # D5 top priority plots
    top_priority_plots = pd.DataFrame(plot_priority["top_priority_plots"])
    top_priority_plots.to_csv(OUTPUT_DIR / "top_priority_plots.csv", index=False)

    print(f"Created {OUTPUT_DIR / 'advisor_kpi_cards.csv'}")
    print(f"Created {OUTPUT_DIR / 'advisor_recommendations_table.csv'}")
    print(f"Created {OUTPUT_DIR / 'advisor_tool_outputs.csv'}")
    print(f"Created {OUTPUT_DIR / 'advisor_guardrails.csv'}")
    print(f"Created {OUTPUT_DIR / 'top_priority_plots.csv'}")


# ============================================================
# Main runner
# ============================================================

def main():
    print("Loading CSV files...")

    sensor_df = read_csv("daily_sensor_readings.csv")
    sensor_df = add_delay_bucket(sensor_df)

    input_costs_df = read_csv("daily_input_costs.csv")
    input_applications_df = read_csv("input_applications.csv")
    season_df = read_csv("season_summary.csv")

    print("Generating advisor outputs...")

    plot_priority = rank_priority_plots(sensor_df)
    response_timing = compare_response_timing(sensor_df)
    precision_roi = summarize_precision_roi(
        input_costs_df, input_applications_df, season_df
    )
    financing_memo = generate_financing_memo_summary(season_df)

    write_json("plot_priority_summary.json", plot_priority)
    write_json("response_timing_summary.json", response_timing)
    write_json("precision_roi_summary.json", precision_roi)
    write_json("financing_memo_summary.json", financing_memo)

    create_advisor_recommendations_md(
        plot_priority, precision_roi, financing_memo
    )

    write_tableau_ready_csvs(
        plot_priority, precision_roi, financing_memo
    )

    print("Done. Advisor outputs created successfully.")


if __name__ == "__main__":
    main()