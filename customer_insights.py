# -*- coding: utf-8 -*-
"""
Olist Customer Insights Dashboard
==================================
A customer-centric analysis of purchasing behavior, loyalty, and activity patterns.

Different from:
  - order_dataset.py : order status management & logistics monitoring
  - EDA.py           : fulfillment & delivery delay analysis

Run: python -m streamlit run customer_insights.py
Keep olist_orders_dataset.csv next to this file.
"""

from datetime import date, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# 1. Page config
# ============================================================

st.set_page_config(
    page_title="Olist Customer Insights",
    page_icon="👥",
    layout="wide",
)

st.title("👥 Olist Customer Insights Dashboard")
st.caption("Understanding customer purchasing behavior, loyalty, and activity patterns")

# ============================================================
# 2. Data loading
# ============================================================

DATA_FILE = Path(__file__).with_name("olist_orders_dataset.csv")
DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


@st.cache_data(show_spinner="Loading customer data...")
def load_data(path: str):
    raw = pd.read_csv(path, dtype={"order_id": "str", "customer_id": "str"})
    required = {"order_id", "customer_id", "order_status", *DATE_COLUMNS}
    if not required.issubset(raw.columns):
        missing = required - raw.columns
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    df = raw.copy()
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    # Derive useful features
    df["purchase_date"] = df["order_purchase_timestamp"].dt.date
    df["purchase_year"] = df["order_purchase_timestamp"].dt.year
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek  # 0=Mon
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["is_cancelled"] = df["order_status"].str.lower().str.strip() == "canceled"
    df["is_delivered"] = df["order_status"].str.lower().str.strip() == "delivered"
    # Delivery duration (only for delivered orders)
    mask = df["order_delivered_customer_date"].notna() & df["order_purchase_timestamp"].notna()
    df["delivery_duration"] = np.where(
        mask,
        (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400,
        np.nan,
    )
    # Time to approval
    mask2 = df["order_approved_at"].notna() & df["order_purchase_timestamp"].notna()
    df["approval_duration"] = np.where(
        mask2,
        (df["order_approved_at"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400,
        np.nan,
    )
    return df


try:
    df = load_data(str(DATA_FILE))
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Could not load order data: {exc}")
    st.info("Place olist_orders_dataset.csv in the same folder as this script.")
    st.stop()

# ============================================================
# 3. Sidebar filters
# ============================================================

st.sidebar.header("🔍 Filters")

# Date range filter
valid_ts = df["order_purchase_timestamp"].dropna()
if valid_ts.empty:
    st.error("No valid purchase timestamps found.")
    st.stop()

min_date = valid_ts.min().date()
max_date = valid_ts.max().date()

date_range = st.sidebar.date_input(
    "Purchase date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Status filter
all_statuses = sorted(df["order_status"].dropna().unique())
status_filter = st.sidebar.multiselect(
    "Order status",
    options=["All"] + all_statuses,
    default=["All"],
)

# Customer filter
unique_customers = df["customer_id"].dropna().unique()
customer_select = st.sidebar.selectbox(
    "View specific customer (optional)",
    options=["All Customers"] + sorted([str(c) for c in unique_customers]),
)

# Apply filters
filtered = df.copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered = filtered[
        (filtered["purchase_date"] >= start_d) & (filtered["purchase_date"] <= end_d)
    ]

if "All" not in status_filter and len(status_filter) > 0:
    filtered = filtered[filtered["order_status"].str.lower().str.strip().isin(status_filter)]

if customer_select != "All Customers":
    filtered = filtered[filtered["customer_id"] == customer_select]

# ============================================================
# 4. Key metrics
# ============================================================

st.subheader("📊 Customer Overview")

if customer_select != "All Customers":
    # Single customer view
    cust_id = customer_select
    cust_orders = filtered[filtered["customer_id"] == cust_id]
    total_cust_orders = len(cust_orders)
    cust_cancel_rate = cust_orders["is_cancelled"].mean() if total_cust_orders else 0
    cust_delivered = cust_orders["is_delivered"].sum()
    cust_avg_delivery = cust_orders["delivery_duration"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders", f"{total_cust_orders}")
    col2.metric("Delivered", f"{cust_delivered}")
    col3.metric("Cancellation Rate", f"{cust_cancel_rate:.1%}" if total_cust_orders else "N/A")
    col4.metric("Avg Delivery Days", f"{cust_avg_delivery:.1f}" if cust_delivered > 0 else "N/A")

    st.caption(f"Customer: {cust_id} | Date range: {date_range}")
else:
    # Aggregate customer view
    total_customers = filtered["customer_id"].nunique()
    total_orders = len(filtered)
    avg_orders_per_cust = total_orders / total_customers if total_customers else 0
    repeat_customers = (
        filtered.groupby("customer_id")
        .size()
        .reset_index(name="cnt")
        .query("cnt > 1")
    )
    repeat_rate = len(repeat_customers) / total_customers if total_customers else 0
    cancelled_rate = filtered["is_cancelled"].mean()
    delivered_mask = filtered["delivery_duration"].notna()
    avg_delivery = filtered.loc[delivered_mask, "delivery_duration"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Customers", f"{total_customers:,}")
    col2.metric("📦 Total Orders", f"{total_orders:,}")
    col3.metric("🔄 Repeat Customer Rate", f"{repeat_rate:.1%}")
    col4.metric("📉 Cancellation Rate", f"{cancelled_rate:.1%}")

    col5, col6 = st.columns(2)
    col5.metric("📊 Avg Orders/Customer", f"{avg_orders_per_cust:.2f}")
    col6.metric("🚚 Avg Delivery Days", f"{avg_delivery:.1f}" if delivered_mask.any() else "N/A")

    st.caption(
        f"Date range: {date_range[0]} – {date_range[1]} | "
        f"Repeat customers: {len(repeat_customers):,} of {total_customers:,}"
    )

st.markdown("---")

# ============================================================
# 5. Customer order count distribution
# ============================================================

if customer_select == "All Customers":
    st.subheader("📈 Customer Order Count Distribution")

    cust_counts = filtered.groupby("customer_id").size().reset_index(name="num_orders")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 4.5), layout="constrained")

    # Cap at 20 for readability
    max_orders_display = min(cust_counts["num_orders"].max(), 20)
    bins = range(0, max_orders_display + 2)
    counts, edges = np.histogram(cust_counts["num_orders"], bins=bins)
    # Re-cap counts
    counts_capped = np.zeros(max_orders_display + 1)
    for i in range(len(counts)):
        if edges[i] < max_orders_display:
            counts_capped[i] = counts[i]
    # Overflow bin
    overflow = cust_counts[cust_counts["num_orders"] > max_orders_display].shape[0]
    counts_capped[-1] += overflow

    colors = ["#27658a"] * (max_orders_display + 1)
    colors[-1] = "#a04e24" if overflow > 0 else "#27658a"

    ax_hist.bar(
        range(max_orders_display + 1),
        counts_capped,
        color=colors,
        edgecolor="white",
        alpha=0.9,
    )
    ax_hist.set_xlabel("Number of Orders per Customer")
    ax_hist.set_ylabel("Number of Customers")
    ax_hist.set_title("Distribution of Order Counts Across Customers")
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.grid(axis="y", alpha=0.2)

    # Add annotation
    median_orders = cust_counts["num_orders"].median()
    ax_hist.axvline(x=median_orders, color="#e74c3c", linestyle="--", linewidth=1.5)
    ax_hist.text(
        median_orders + 0.3,
        ax_hist.get_ylim()[1] * 0.9,
        f"Median: {median_orders:.1f}",
        color="#e74c3c",
        fontsize=10,
    )

    st.pyplot(fig_hist)
    plt.close(fig_hist)

    # Summary stats
    st.caption(
        f"Customers with 1 order: {(cust_counts['num_orders'] == 1).sum():,} | "
        f"2 orders: {(cust_counts['num_orders'] == 2).sum():,} | "
        f"3+ orders: {(cust_counts['num_orders'] >= 3).sum():,} | "
        f"Max orders (capped at {max_orders_display}): "
        f"{cust_counts[cust_counts['num_orders'] <= max_orders_display]['num_orders'].max()}"
    )

    st.markdown("---")

# ============================================================
# 6. Monthly active customers & purchase patterns
# ============================================================

st.subheader("📅 Customer Activity Over Time")

monthly_cust = (
    filtered.dropna(subset=["order_purchase_timestamp"])
    .copy()
)
monthly_cust["month"] = monthly_cust["order_purchase_timestamp"].dt.to_period("M")

monthly_active = (
    monthly_cust.groupby("month")
    .agg(
        unique_customers=("customer_id", "nunique"),
        total_orders=("order_id", "nunique"),
    )
    .reindex(pd.period_range(monthly_cust["month"].min(), monthly_cust["month"].max(), freq="M"))
)
# Fill NaN months with 0 for orders but keep NaN for customers
monthly_active["total_orders"] = monthly_active["total_orders"].fillna(0)

fig_month, ax_month = plt.subplots(figsize=(12, 4.5), layout="constrained")

months_short = [str(m).replace("-0", "-") for m in monthly_active.index]
positions = list(range(len(monthly_active)))

ax_month_twin = ax_month.twinx()
ax_month.bar(
    positions,
    monthly_active["unique_customers"].fillna(0),
    color="#27658a",
    alpha=0.7,
    label="Active Customers",
    width=0.6,
)
ax_month_twin.plot(
    positions,
    monthly_active["total_orders"],
    color="#a04e24",
    marker="o",
    linewidth=2,
    label="Total Orders",
)

# Ticks
step = max(1, (len(months_short) + 8) // 10)
ticks = positions[::step]
ax_month.set_xticks(ticks, [months_short[i] for i in ticks], rotation=45, ha="right")
ax_month.set_xlabel("Month")
ax_month.set_ylabel("Active Customers", color="#27658a")
ax_month_twin.set_ylabel("Total Orders", color="#a04e24")

# Legends
lines1, labels1 = ax_month.get_legend_handles_labels()
lines2, labels2 = ax_month_twin.get_legend_handles_labels()
ax_month.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

ax_month.spines["top"].set_visible(False)
ax_month.grid(axis="y", alpha=0.2)

st.pyplot(fig_month)
plt.close(fig_month)

st.caption("Blue bars = unique customers placing orders that month; Orange line = total order count.")

st.markdown("---")

# ============================================================
# 7. Day-of-week purchase patterns
# ============================================================

st.subheader("🗓️ Purchase Patterns by Day of Week")

dow_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}

dow_data = filtered.groupby("purchase_dayofweek").agg(
    order_count=("order_id", "nunique"),
    unique_customers=("customer_id", "nunique"),
).reindex(range(7))

dow_data["day_name"] = dow_data.index.map(dow_map)

fig_dow, ax_dow = plt.subplots(figsize=(10, 4.5), layout="constrained")

x = list(range(7))
bars1 = ax_dow.bar(
    [i - 0.2 for i in x],
    dow_data["order_count"],
    width=0.35,
    color="#27658a",
    alpha=0.85,
    label="Orders",
    edgecolor="white",
)
bars2 = ax_dow.bar(
    [i + 0.2 for i in x],
    dow_data["unique_customers"],
    width=0.35,
    color="#a04e24",
    alpha=0.85,
    label="Customers",
    edgecolor="white",
)

ax_dow.set_xticks(x)
ax_dow.set_xticklabels(dow_data["day_name"], rotation=30, ha="right")
ax_dow.set_ylabel("Count")
ax_dow.set_title("Order & Customer Activity by Day of Week")
ax_dow.legend()
ax_dow.spines["top"].set_visible(False)
ax_dow.spines["right"].set_visible(False)
ax_dow.grid(axis="y", alpha=0.2)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    if height > 0:
        ax_dow.text(bar.get_x() + bar.get_width() / 2., height, f"{int(height):,}",
                    ha="center", va="bottom", fontsize=9)
for bar in bars2:
    height = bar.get_height()
    if height > 0:
        ax_dow.text(bar.get_x() + bar.get_width() / 2., height, f"{int(height):,}",
                    ha="center", va="bottom", fontsize=9)

st.pyplot(fig_dow)
plt.close(fig_dow)

st.markdown("---")

# ============================================================
# 8. Hourly purchase distribution (if data has time component)
# ============================================================

st.subheader("🕐 Purchase Activity by Hour of Day")

hour_data = filtered.groupby("purchase_hour").size().reset_index(name="count")

fig_hour, ax_hour = plt.subplots(figsize=(12, 4.5), layout="constrained")

# Color by peak vs off-peak
peak_hour = hour_data.loc[hour_data["count"].idxmax(), "hour"] if len(hour_data) else 0
colors_hour = ["#a04e24" if h == peak_hour else "#27658a" for h in hour_data["hour"]]

ax_hour.bar(hour_data["hour"], hour_data["count"], color=colors_hour, alpha=0.85, edgecolor="white")
ax_hour.set_xlabel("Hour of Day")
ax_hour.set_ylabel("Number of Orders")
ax_hour.set_title(f"Order Distribution by Hour (Peak: {peak_hour:02d}:00)")
ax_hour.spines["top"].set_visible(False)
ax_hour.spines["right"].set_visible(False)
ax_hour.grid(axis="y", alpha=0.2)

st.pyplot(fig_hour)
plt.close(fig_hour)

st.markdown("---")

# ============================================================
# 9. Customer cancellation rate comparison
# ============================================================

st.subheader("📊 Order Status Breakdown by Customer")

if customer_select == "All Customers":
    # Show status breakdown for top customers by order count
    top_n = 15
    top_customers = (
        filtered.groupby("customer_id")
        .size()
        .reset_index(name="total")
        .sort_values("total", ascending=False)
        .head(top_n)
    )

    status_by_cust = filtered[filtered["customer_id"].isin(top_customers["customer_id"])]
    status_pivot = status_by_cust.groupby(["customer_id", "order_status"]).size().unstack(fill_value=0)

    # Reorder by total
    cust_order = top_customers.set_index("customer_id")["total"].index
    common_statuses = [c for c in status_pivot.columns if c in cust_order]
    status_pivot = status_pivot.reindex(cust_order)

    # Shorten customer IDs for readability
    status_pivot.index = [str(c)[:12] + "..." if len(str(c)) > 12 else str(c) for c in status_pivot.index]

    fig_status, ax_status = plt.subplots(figsize=(11, 5), layout="constrained")
    status_pivot.plot(kind="bar", stacked=True, ax=ax_status, width=0.85)
    ax_status.set_xlabel("Customer ID")
    ax_status.set_ylabel("Number of Orders")
    ax_status.set_title(f"Order Status Breakdown — Top {top_n} Customers by Order Count")
    ax_status.legend(title="Status", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax_status.spines["top"].set_visible(False)
    ax_status.spines["right"].set_visible(False)
    ax_status.tick_params(axis="x", rotation=45, labelsize=8)
    st.pyplot(fig_status)
    plt.close(fig_status)
else:
    # Single customer status breakdown
    cust_status = filtered.groupby("order_status").size().reset_index(name="count")
    fig_cs, ax_cs = plt.subplots(figsize=(8, 4), layout="constrained")
    ax_cs.pie(
        cust_status["count"],
        labels=cust_status["order_status"],
        autopct="%1.1f%%",
        colors=["#27658a", "#a04e24", "#27ae60", "#f39c12", "#95a5a6"],
        startangle=90,
    )
    ax_cs.set_title("Order Status Breakdown for Selected Customer")
    st.pyplot(fig_cs)
    plt.close(fig_cs)

st.markdown("---")

# ============================================================
# 10. Customer delivery time analysis
# ============================================================

st.subheader("🚚 Delivery Duration Analysis")

delivered_data = filtered[filtered["delivery_duration"].notna()].copy()

if len(delivered_data) > 0:
    if customer_select == "All Customers":
        # Delivery duration by month
        delivered_data["month"] = delivered_data["order_purchase_timestamp"].dt.to_period("M")

        monthly_delivery = (
            delivered_data.groupby("month")
            .agg(
                avg_days=("delivery_duration", "mean"),
                median_days=("delivery_duration", "median"),
                count=("order_id", "nunique"),
            )
            .reindex(pd.period_range(delivered_data["month"].min(), delivered_data["month"].max(), freq="M"))
        )

        fig_del, ax_del = plt.subplots(figsize=(12, 4.5), layout="constrained")
        months_s = [str(m) for m in monthly_delivery.index]

        ax_del.plot(range(len(monthly_delivery)), monthly_delivery["avg_days"],
                     color="#a04e24", marker="o", linewidth=2, label="Mean")
        ax_del.plot(range(len(monthly_delivery)), monthly_delivery["median_days"],
                     color="#27658a", marker="s", linewidth=2, label="Median")
        ax_del.fill_between(
            range(len(monthly_delivery)),
            monthly_delivery["median_days"],
            monthly_delivery["avg_days"],
            alpha=0.15,
            color="#a04e24",
        )

        step = max(1, (len(months_s) + 8) // 10)
        ticks = list(range(0, len(months_s), step))
        ax_del.set_xticks(ticks, [months_s[i] for i in ticks], rotation=45, ha="right")
        ax_del.set_xlabel("Month")
        ax_del.set_ylabel("Delivery Duration (days)")
        ax_del.set_title("Average & Median Delivery Duration Over Time")
        ax_del.legend()
        ax_del.spines["top"].set_visible(False)
        ax_del.grid(axis="y", alpha=0.2)

        st.pyplot(fig_del)
        plt.close(fig_del)

        # Delivery duration histogram
        fig_hist_del, ax_hist_del = plt.subplots(figsize=(10, 4.5), layout="constrained")
        ax_hist_del.hist(delivered_data["delivery_duration"], bins=40, color="#27658a",
                          alpha=0.85, edgecolor="white", density=True)
        ax_hist_del.axvline(delivered_data["delivery_duration"].mean(), color="#a04e24",
                            linestyle="--", linewidth=2, label=f"Mean: {delivered_data['delivery_duration'].mean():.1f} days")
        ax_hist_del.axvline(delivered_data["delivery_duration"].median(), color="#27ae60",
                            linestyle="--", linewidth=2, label=f"Median: {delivered_data['delivery_duration'].median():.1f} days")
        ax_hist_del.set_xlabel("Delivery Duration (days)")
        ax_hist_del.set_ylabel("Density")
        ax_hist_del.set_title("Delivery Duration Distribution")
        ax_hist_del.legend()
        ax_hist_del.spines["top"].set_visible(False)
        ax_hist_del.spines["right"].set_visible(False)
        ax_hist_del.grid(axis="y", alpha=0.2)

        st.pyplot(fig_hist_del)
        plt.close(fig_hist_del)
    else:
        # Single customer delivery times
        cust_delivered = delivered_data[delivered_data["customer_id"] == customer_select]
        if len(cust_delivered) > 0:
            fig_cd, ax_cd = plt.subplots(figsize=(10, 4), layout="constrained")
            ax_cd.hist(cust_delivered["delivery_duration"], bins=15, color="#27658a",
                        alpha=0.85, edgecolor="white")
            ax_cd.axvline(cust_delivered["delivery_duration"].mean(), color="#a04e24",
                           linestyle="--", linewidth=2, label=f"Mean: {cust_delivered['delivery_duration'].mean():.1f} days")
            ax_cd.set_xlabel("Delivery Duration (days)")
            ax_cd.set_ylabel("Count")
            ax_cd.set_title(f"Delivery Duration for {customer_select[:12]}")
            ax_cd.legend()
            ax_cd.spines["top"].set_visible(False)
            ax_cd.spines["right"].set_visible(False)
            ax_cd.grid(axis="y", alpha=0.2)
            st.pyplot(fig_cd)
            plt.close(fig_cd)

    st.caption(
        f"Delivered orders in range: {len(delivered_data):,} | "
        f"Mean: {delivered_data['delivery_duration'].mean():.1f} days | "
        f"Median: {delivered_data['delivery_duration'].median():.1f} days | "
        f"Max: {delivered_data['delivery_duration'].max():.1f} days"
    )
else:
    st.info("No delivered orders with valid timestamps in the selected range.")

st.markdown("---")

# ============================================================
# 11. Customer order details table
# ============================================================

st.subheader("📋 Order Details")

st.write(f"Showing **{len(filtered):,}** orders under current filters.")

# Prepare display dataframe
display_df = filtered[[
    "order_id", "customer_id", "order_status",
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date", "delivery_duration", "is_cancelled",
]].copy()

rename_map = {
    "order_id": "Order ID",
    "customer_id": "Customer ID",
    "order_status": "Status",
    "order_purchase_timestamp": "Purchase Time",
    "order_approved_at": "Approval Time",
    "order_delivered_carrier_date": "Handed to Carrier",
    "order_delivered_customer_date": "Delivered to Customer",
    "order_estimated_delivery_date": "Estimated Delivery",
    "delivery_duration": "Delivery Days",
    "is_cancelled": "Cancelled",
}

display_df = display_df.rename(columns=rename_map)
# Format boolean
if "Cancelled" in display_df.columns:
    display_df["Cancelled"] = display_df["Cancelled"].map({True: "Yes", False: "No"})
# Format NaN
display_df = display_df.fillna("")

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ============================================================
# 12. Export
# ============================================================

st.subheader("📥 Export")

csv_data = display_df.to_csv(index=False, encoding="utf-8-sig")
st.download_button(
    label="Download filtered orders as CSV",
    data=csv_data,
    file_name=f"olist_customer_insights_{date.today()}.csv",
    mime="text/csv",
)

# ============================================================
# 13. Footer
# ============================================================

st.markdown("---")
st.caption("数据来源：Olist Brazilian E-Commerce Public Dataset (Kaggle)")
st.caption(
    "Note: This dashboard analyzes customer purchasing behavior from historical order data. "
    "Metrics are computed on the filtered subset only."
)
