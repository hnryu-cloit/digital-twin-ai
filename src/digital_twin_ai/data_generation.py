"""
data_generation.py - Logic for generating dummy Excel data and synthetic customer data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import datetime

# Common reference values from 00_generate_dummy_excel.py
AP2_REGIONS = ["SEA", "SEAU", "SEF", "SEG", "SEAS", "SEASA", "KOREA", "SECA", "SWA"]
AP2_WEIGHTS = [0.18, 0.10, 0.15, 0.08, 0.06, 0.07, 0.12, 0.08, 0.16]

COUNTRY_BY_AP2 = {
    "SEA": ["United States"],
    "SEAU": ["Australia", "Singapore", "Thailand", "Vietnam", "Malaysia"],
    "SEF": ["France", "Spain", "Italy", "Netherlands", "Belgium"],
    "SEG": ["Germany", "Switzerland"],
    "SEAS": ["Austria", "Poland", "Czech Republic"],
    "SEASA": ["Brazil", "Mexico", "Argentina", "Colombia"],
    "KOREA": ["South Korea"],
    "SECA": ["Canada"],
    "SWA": ["India", "Pakistan", "Bangladesh"],
}

CNTY_CD_BY_AP2 = {
    "SEA": "USA", "SEAU": "AUS", "SEF": "FRA", "SEG": "DEU",
    "SEAS": "AUT", "SEASA": "BRA", "KOREA": "KOR", "SECA": "CAN", "SWA": "IND",
}

CNTY_CD2_BY_AP2 = {
    "SEA": "US", "SEAU": "AU", "SEF": "FR", "SEG": "DE",
    "SEAS": "AT", "SEASA": "BR", "KOREA": "KR", "SECA": "CA", "SWA": "IN",
}

GC_BY_AP2 = {
    "SEA": "N.America", "SEAU": "S.E.Asia", "SEF": "Europe",
    "SEG": "Europe", "SEAS": "Europe", "SEASA": "L.America",
    "KOREA": "KOREA", "SECA": "N.America", "SWA": "S.W.Asia",
}

CURRENCIES_BY_AP2 = {
    "SEA": ("USD", 1.0), "SEAU": ("AUD", 1.52),
    "SEF": ("EUR", 0.92), "SEG": ("EUR", 0.92),
    "SEAS": ("EUR", 0.92), "SEASA": ("BRL", 5.0),
    "KOREA": ("KRW", 1320.0), "SECA": ("CAD", 1.36),
    "SWA": ("INR", 83.0),
}

ACTIVENESS_LEVELS = ["Active (1M)", "Active (3M)", "Active (6M)", "Active (12M)", "Inactive"]
ACTIVENESS_WEIGHTS = [0.25, 0.20, 0.18, 0.15, 0.22]

VOYAGER_SEGMENTS = [
    "Gamer", "Premium Buyer", "Health Enthusiast", "Smart Home User",
    "Budget Shopper", "Early Adopter", "Loyal Customer", "Fashion Forward",
]

PRODUCTS_BY_CAT = {
    "HHP": [
        ("Galaxy S25 Ultra", "S", "SM-S938B", "SMART", "Smartphones", "Galaxy_S", "Y", "6.9_UB"),
        ("Galaxy S25+", "S", "SM-S936B", "SMART", "Smartphones", "Galaxy_S", "Y", "6.7_UB"),
        ("Galaxy A55", "A", "SM-A556B", "SMART", "Smartphones", "Galaxy_A", "Y", "6.6_UB"),
        ("Galaxy A35", "A", "SM-A356B", "SMART", "Smartphones", "Galaxy_A", "Y", "6.6_UB"),
        ("Galaxy Z Fold6", "Z Fold", "SM-F956B", "FOLD", "Foldables", "Galaxy_Z_Fold", "Y", "7.6_UB"),
        ("Galaxy Z Flip6", "Z Flip", "SM-F741B", "FLIP", "Foldables", "Galaxy_Z_Flip", "Y", "6.7_UB"),
    ],
    "WEARABLE": [
        ("Galaxy Watch 7", "Watch7", "SM-L300N", "WATCH", "Watches", "Galaxy_Watch", "Y", None),
        ("Galaxy Watch Ultra", "Watch Ultra", "SM-L705N", "WATCH", "Watches", "Galaxy_Watch", "Y", None),
        ("Galaxy Buds3 Pro", "Buds3", "SM-R630N", "BUDS", "Earbuds", "Galaxy_Buds", "N", None),
        ("Galaxy Ring", "Ring", "SM-Q501N", "RING", "Wearables", "Galaxy_Ring", "N", None),
    ],
    "TV": [
        ("Neo QLED 8K 85\"", "Neo QLED", "QA85QN900D", "QLED", "TVs", "Neo_QLED", "Y", "85\""),
        ("OLED 77\"", "OLED", "QA77S90D", "OLED", "TVs", "OLED", "Y", "77\""),
        ("Crystal UHD 65\"", "Crystal UHD", "UA65AU8000", "UHD", "TVs", "Crystal_UHD", "Y", "65\""),
    ],
    "TABLET": [
        ("Galaxy Tab S9 Ultra", "Tab S9", "SM-X916B", "TABLET", "Tablets", "Galaxy_Tab_S", "Y", "14.6\""),
        ("Galaxy Tab S9 FE", "Tab S9 FE", "SM-X516B", "TABLET", "Tablets", "Galaxy_Tab_S", "Y", "10.9\""),
    ],
    "PC": [
        ("Galaxy Book4 Pro", "Book4", "NP960XGK", "LAPTOP", "Laptops", "Galaxy_Book", "Y", "16\""),
        ("Galaxy Book4 360", "Book4", "NP730QGK", "LAPTOP", "Laptops", "Galaxy_Book", "Y", "13.3\""),
    ],
    "AUDIO": [
        ("Galaxy Buds2 Pro", "Buds2", "SM-R510N", "BUDS", "Earbuds", "Galaxy_Buds", "N", None),
        ("Sound Bar Q990D", "Sound Bar", "HW-Q990D", "SOUNDBAR", "Soundbars", "Sound_Bar", "Y", None),
    ],
    "DA": [
        ("Bespoke AI Washer", "Bespoke", "WF25BB6900H", "WASHER", "Appliances", "Bespoke_WM", "Y", None),
        ("Bespoke French Door Fridge", "Bespoke", "RF29BB6200A", "FRIDGE", "Appliances", "Bespoke_REF", "Y", None),
    ],
}

PD_CATEGORIES = list(PRODUCTS_BY_CAT.keys())
PD_CATEGORY_WEIGHTS = [0.40, 0.15, 0.12, 0.10, 0.08, 0.08, 0.07]

DIVISION_BY_CAT = {
    "HHP": "MX", "WEARABLE": "MX", "TABLET": "MX", "PC": "MX",
    "TV": "VD", "AUDIO": "VD", "DA": "DA",
}
GROUP_BY_CAT = {
    "HHP": "MOBILE", "WEARABLE": "ME", "TABLET": "TABLET", "PC": "PC",
    "TV": "TV", "AUDIO": "AUDIO", "DA": "DA",
}

STORE_TYPES = ["D2C", "EPP", "Retail", "Carrier", "Operator"]
STORE_WEIGHTS = [0.30, 0.15, 0.30, 0.15, 0.10]

SAMSUNG_APPS = [
    ("com.samsung.android.health", "Samsung Health", "HEALTH"),
    ("com.samsung.android.app.galaxyfinder", "Galaxy Store", "UTILITY"),
    ("com.samsung.android.messaging", "Samsung Messages", "UTILITY"),
    ("com.samsung.android.email.provider", "Samsung Email", "PRODUCTIVITY"),
    ("com.samsung.android.spay", "Samsung Pay", "FINANCE"),
    ("com.samsung.android.smartthings", "SmartThings", "UTILITY"),
]

NON_SAMSUNG_APPS = [
    ("com.facebook.katana", "Facebook", "SOCIAL"),
    ("com.instagram.android", "Instagram", "SOCIAL"),
    ("com.google.android.youtube", "YouTube", "ENTERTAINMENT"),
    ("com.supercell.hayday", "Hay Day", "GAME"),
    ("com.king.candycrushsaga", "Candy Crush Saga", "GAME"),
    ("com.netflix.mediaclient", "Netflix", "ENTERTAINMENT"),
    ("com.spotify.music", "Spotify", "ENTERTAINMENT"),
    ("com.amazon.mShop.android.shopping", "Amazon Shopping", "SHOPPING"),
    ("com.google.android.gm", "Gmail", "PRODUCTIVITY"),
    ("com.whatsapp", "WhatsApp", "SOCIAL"),
    ("com.tiktok.android", "TikTok", "SOCIAL"),
    ("com.twitter.android", "X (Twitter)", "SOCIAL"),
    ("com.kakao.talk", "KakaoTalk", "SOCIAL"),
    ("com.garena.freefireth", "Free Fire", "GAME"),
    ("com.riotgames.league.wildrift", "Wild Rift", "GAME"),
    ("com.paypal.android.p2pmobile", "PayPal", "FINANCE"),
    ("com.google.android.apps.maps", "Google Maps", "UTILITY"),
]

INTEREST_CATEGORIES = [
    "ART & CUSTOMIZED", "BANKING & FINANCE", "ENTERTAINMENT & MEDIA",
    "FASHION & BEAUTY", "FITNESS & WELLNESS", "FOOD & BEVERAGE",
    "GAMING", "SMART HOME & IOT", "SPORTS", "TRAVEL & LEISURE",
]


class DataGenerator:
    """Class for generating synthetic customer data."""

    def __init__(self, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)

    def generate_dummy_excel(self, n_customers: int, output_path: str) -> None:
        """Generate dummy Excel data for testing purposes."""
        print(f"Generating dummy Excel data... (N={n_customers})")

        demo = self._generate_demo(n_customers)
        clv = self._generate_clv(demo)
        purch = self._generate_purchase(demo)
        owned = self._generate_owned_devices(demo)
        app = self._generate_app_usage(demo)
        inter = self._generate_interests(demo)
        rewards = self._generate_rewards(demo)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            self._write_sheet(writer, demo, "Demo")
            self._write_sheet(writer, clv, "CLV")
            self._write_sheet(writer, purch, "구매")
            self._write_sheet(writer, owned, "보유")
            self._write_sheet(writer, app, "앱사용")
            self._write_sheet(writer, inter, "관심사")
            self._write_sheet(writer, rewards, "리워즈")

    def _write_sheet(self, writer: pd.ExcelWriter, df: pd.DataFrame, sheet_name: str) -> None:
        df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)

    def _generate_demo(self, n: int) -> pd.DataFrame:
        ap2 = self.rng.choice(AP2_REGIONS, n, p=AP2_WEIGHTS)
        countries = [self.rng.choice(COUNTRY_BY_AP2[a]) for a in ap2]

        age_grp = self.rng.choice([0, 1, 2], n, p=[0.35, 0.40, 0.25])
        age = np.where(
            age_grp == 0, self.rng.integers(18, 30, n),
            np.where(age_grp == 1, self.rng.integers(30, 50, n),
                     self.rng.integers(50, 70, n))
        ).clip(18, 75)

        voyager = [
            str(list(self.rng.choice(VOYAGER_SEGMENTS, int(self.rng.integers(1, 4)), replace=False)))
            for _ in range(n)
        ]

        return pd.DataFrame({
            "index": range(1, n + 1),
            "sa_activeness": self.rng.choice(ACTIVENESS_LEVELS, n, p=ACTIVENESS_WEIGHTS),
            "usr_age": age,
            "usr_gndr": self.rng.choice(["M", "F"], n, p=[0.52, 0.48]),
            "usr_cnty_name": countries,
            "usr_cnty_ap2": ap2,
            "relation_all_cnt": self.rng.integers(1, 5, n),
            "relation_family_group_cnt": self.rng.integers(0, 3, n),
            "relation_estimated_cnt": self.rng.integers(1, 4, n),
            "voyager_segment": voyager,
        })

    def _generate_clv(self, demo: pd.DataFrame) -> pd.DataFrame:
        n = len(demo)
        age_arr = demo["usr_age"].values
        active_arr = demo["sa_activeness"].str.startswith("Active").values.astype(float)

        age_factor = 1 - (age_arr - 18) / 100
        active_factor = active_arr * 0.2
        retention = np.clip(age_factor * 0.5 + active_factor + self.rng.uniform(0.2, 0.5, n), 0.2, 0.99)

        ltv = np.clip(self.rng.lognormal(7.5, 1.2, n), 50, 50_000)
        pchs_cnt = self.rng.integers(1, 8, n)
        prem_cnt = np.minimum(self.rng.integers(0, 4, n), pchs_cnt)

        repurch_p = np.clip(retention * 0.8, 0.1, 0.9)
        health_p = np.clip(0.35 + age_factor * 0.1, 0.1, 0.8)
        wallet_p = np.clip(0.25 + active_factor * 0.3, 0.1, 0.8)
        smartthings_p = np.clip(0.20 + prem_cnt / 10, 0.1, 0.7)

        first_dt = pd.to_datetime("2018-01-01") + pd.to_timedelta(self.rng.integers(0, 2000, n), unit="D")
        raw_last = first_dt + pd.to_timedelta(self.rng.integers(30, 1800, n), unit="D")
        cap = pd.Timestamp("2025-03-01")
        last_dt = pd.DatetimeIndex([min(d, cap) for d in raw_last])

        wallet_dt = pd.to_datetime("2021-01-01") + pd.to_timedelta(self.rng.integers(0, 1200, n), unit="D")
        health_dt = pd.to_datetime("2019-01-01") + pd.to_timedelta(self.rng.integers(0, 2000, n), unit="D")
        smartth_dt = pd.array([pd.NaT] * n, dtype="datetime64[ns]")

        product_map = self.rng.choice(["HHP", "WATCH", "TV", "TABLET"], n, p=[0.5, 0.2, 0.2, 0.1])
        div_fin = np.where(np.isin(product_map, ["HHP", "WATCH", "TABLET"]), "MX", "VD")

        return pd.DataFrame({
            "index": demo["index"].values,
            "bs_date": "2024-12-27",
            "gc": [GC_BY_AP2[a] for a in demo["usr_cnty_ap2"]],
            "cnty_cd": [CNTY_CD_BY_AP2[a] for a in demo["usr_cnty_ap2"]],
            "subsidiary": demo["usr_cnty_ap2"].values,
            "gender": demo["usr_gndr"].values,
            "age": age_arr,
            "age_band": pd.cut(age_arr, bins=[0, 29, 39, 49, 59, 100],
                               labels=["20s", "30s", "40s", "50s", "60s+"]).astype(str),
            "product_mapping4": product_map,
            "division_fin": div_fin,
            "retention_score": retention.round(3),
            "retention_adj": np.clip(retention * self.rng.uniform(0.8, 1.0, n), 0.1, 0.99).round(3),
            "val_p": np.clip(self.rng.lognormal(6.5, 1.0, n), 50, 20_000).round(2),
            "pchs_cnt": pchs_cnt,
            "div_cnt": self.rng.integers(1, 4, n),
            "prod_cnt": self.rng.integers(1, 6, n),
            "prod_cnt_bydiv": self.rng.integers(1, 4, n),
            "prd_cyc_adj_y": self.rng.uniform(0.5, 4.0, n).round(2),
            "pchs_cyc_org_y": self.rng.uniform(0.5, 4.0, n).round(2),
            "val_f_r": (ltv * self.rng.uniform(0.5, 3.0, n)).round(2),
            "ltv_r": ltv.round(2),
            "first_reg_dt": first_dt.strftime("%Y-%m-%d"),
            "last_reg_dt": last_dt.strftime("%Y-%m-%d"),
            "cum_repchs_flg": (self.rng.random(n) < repurch_p).astype(int),
            "new_repchs_flg": self.rng.integers(0, 2, n),
            "cum_prod_repchs_flg": self.rng.integers(0, 2, n),
            "new_prod_repchs_flg": self.rng.integers(0, 2, n),
            "cum_upsell_flg": self.rng.integers(0, 2, n),
            "new_upsell_flg": self.rng.integers(0, 2, n),
            "d2c_flg": self.rng.integers(0, 2, n),
            "rr_mapping": "GUID",
            "age_null_yn": 0,
            "st_act_flg": (self.rng.random(n) < smartthings_p).astype(int),
            "st_div_flg": self.rng.integers(0, 2, n),
            "reg_diff": np.maximum((last_dt - first_dt).days, 0),
            "flag_hhp_only": self.rng.random(n) < 0.30,
            "flag_tv_only": self.rng.random(n) < 0.10,
            "samsungwallet_flag": self.rng.random(n) < wallet_p,
            "samsungwallet_first_dt": wallet_dt,
            "samsunghealth_flag": self.rng.random(n) < health_p,
            "samsunghealth_first_dt": health_dt,
            "smartthings_flag": self.rng.random(n) < smartthings_p,
            "smartthings_first_dt": smartth_dt,
            "premium_cnt": prem_cnt,
        })

    def _generate_purchase(self, demo: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, cust in demo.iterrows():
            ap2 = cust["usr_cnty_ap2"]
            currency, fx = CURRENCIES_BY_AP2.get(ap2, ("USD", 1.0))
            country = cust["usr_cnty_name"]
            n_orders = int(self.rng.integers(1, 6))

            for _ in range(n_orders):
                cat = self.rng.choice(PD_CATEGORIES, p=PD_CATEGORY_WEIGHTS)
                prod_list = PRODUCTS_BY_CAT[cat]
                pd_name, pd_series, sku, pd_type, pd_rec_type, pd_rec_sub_type, pd_smart, pd_screen \
                    = prod_list[self.rng.integers(0, len(prod_list))]

                is_prem = int(self.rng.random() < 0.35)
                base_usd = float(np.clip(
                    self.rng.lognormal(5.5, 0.8) if is_prem else self.rng.lognormal(4.8, 0.7), 30, 3000
                ))
                disc_usd = base_usd * self.rng.uniform(0, 0.15)
                tradein_usd = base_usd * self.rng.uniform(0, 0.10) if self.rng.random() < 0.20 else None
                rewards_usd = base_usd * self.rng.uniform(0, 0.05) if self.rng.random() < 0.30 else None
                sale_usd = base_usd - disc_usd - (tradein_usd or 0) - (rewards_usd or 0)
                sale_usd = max(sale_usd, 1.0)

                order_dt = pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(self.rng.integers(0, 2557)))
                color = self.rng.choice(["BLACK", "WHITE", "SILVER", "BLUE", "CREAM", "PHANTOM_BLACK"])
                storage = self.rng.choice(["128GB", "256GB", "512GB", None], p=[0.35, 0.35, 0.20, 0.10]) \
                    if cat in ["HHP", "TABLET", "PC"] else None

                rows.append({
                    "index": cust["index"],
                    "purc_cnty_ap2": ap2,
                    "purc_cnty_name": country,
                    "order_id": f"{ap2[:2]}{order_dt.strftime('%y%m%d')}-{int(self.rng.integers(10_000_000, 99_999_999))}",
                    "order_date": order_dt.strftime("%Y-%m-%d"),
                    "store_type": self.rng.choice(STORE_TYPES, p=STORE_WEIGHTS),
                    "site_name": "Samsung.com",
                    "source_app": self.rng.choice(["Mobile", "PC"], p=[0.6, 0.4]),
                    "order_entries_oe_sku": sku,
                    "order_entries_oe_name": pd_name,
                    "sale_qty": 1,
                    "sale_amt_local": round(sale_usd * fx, 2),
                    "price_base_local": round(base_usd * fx, 2),
                    "price_discount_all_local": round(disc_usd * fx, 2),
                    "price_discount_tradein_local": round(tradein_usd * fx, 2) if tradein_usd else None,
                    "price_discount_rewards_local": round(rewards_usd * fx, 2) if rewards_usd else None,
                    "currency": currency,
                    "exchange_rate": fx,
                    "sale_amt_usd": round(sale_usd, 2),
                    "price_base_usd": round(base_usd, 2),
                    "price_discount_all_usd": round(disc_usd, 2),
                    "price_discount_tradein_usd": round(tradein_usd, 2) if tradein_usd else None,
                    "price_discount_rewards_usd": round(rewards_usd, 2) if rewards_usd else None,
                    "pd_division": DIVISION_BY_CAT[cat],
                    "pd_group": GROUP_BY_CAT[cat],
                    "pd_category": cat,
                    "pd_type": pd_type,
                    "pd_rec_type": pd_rec_type,
                    "pd_rec_sub_type": pd_rec_sub_type,
                    "pd_series": pd_series,
                    "pd_name": pd_name,
                    "pd_color": color,
                    "pd_size": None,
                    "pd_smart": pd_smart,
                    "refurbished": "N",
                    "pd_screen": pd_screen,
                    "pd_storage": storage,
                    "pd_marketing_name": pd_name,
                    "pd_description": f"{cat},{sku},{color}",
                    "pd_mkt_attb01": pd_smart,
                    "pd_mkt_attb02": pd_series,
                    "pd_mkt_attb03": pd_name,
                    "data_source": "hybris",
                    "sellin_price": round(base_usd * 0.65, 2),
                    "premium_flg": is_prem,
                })

        return pd.DataFrame(rows)

    def _generate_owned_devices(self, demo: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, cust in demo.iterrows():
            n_dev = int(self.rng.integers(1, 5))
            for dvc_idx in range(1, n_dev + 1):
                cat = self.rng.choice(PD_CATEGORIES, p=PD_CATEGORY_WEIGHTS)
                prod_list = PRODUCTS_BY_CAT[cat]
                pd_name, pd_series, sku, pd_type, pd_rec_type, pd_rec_sub_type, pd_smart, _ \
                    = prod_list[self.rng.integers(0, len(prod_list))]

                reg_dt = pd.Timestamp("2017-01-01") + pd.Timedelta(days=int(self.rng.integers(0, 2920)))
                last_reg = reg_dt + pd.Timedelta(days=int(self.rng.integers(0, 300)))
                last_act = last_reg + pd.Timedelta(days=int(self.rng.integers(0, 90))) \
                    if self.rng.random() < 0.70 else None

                base_usd = float(np.clip(self.rng.lognormal(5.0, 0.8), 30, 3000))

                rows.append({
                    "index": cust["index"],
                    "dvc_index": dvc_idx,
                    "usr_is_last_owner": self.rng.choice(["Y", "N"], p=[0.85, 0.15]),
                    "dvc_division": DIVISION_BY_CAT[cat],
                    "dvc_group": GROUP_BY_CAT[cat],
                    "dvc_category": cat,
                    "dvc_type": {"HHP": "Smart Phone", "WEARABLE": "Wearable", "TV": "TV",
                                 "TABLET": "Tablet", "PC": "Laptop", "AUDIO": "Audio", "DA": "Appliance"}[cat],
                    "dvc_pd_type": pd_type,
                    "dvc_rec_type": pd_rec_type,
                    "dvc_rec_sub_type": pd_rec_sub_type,
                    "dvc_mdl_id": sku,
                    "dvc_mdl_nm": pd_name,
                    "dvc_sku": sku,
                    "dvc_series": pd_series,
                    "dvc_attb_color": self.rng.choice(["BLACK", "WHITE", "SILVER", "BLUE", "CREAM"]),
                    "dvc_attb_size": None,
                    "dvc_attb_smart": pd_smart,
                    "dvc_attb_etc": None,
                    "dvc_acc_yn": "N",
                    "dvc_is_cellular": self.rng.random() < 0.70,
                    "dvc_is_wifi": True,
                    "dvc_is_bluetooth": True,
                    "dvc_is_secondhand": self.rng.choice(["N", "Y"], p=[0.90, 0.10]),
                    "dvc_manufacturer": "Samsung",
                    "dvc_samsung_flag": "Y",
                    "refurbished": "N",
                    "dvc_reg_date": reg_dt.strftime("%Y-%m-%d"),
                    "dvc_first_reg_date": reg_dt.strftime("%Y-%m-%d"),
                    "dvc_last_reg_date": last_reg.strftime("%Y-%m-%d"),
                    "dvc_last_active_date": last_act.strftime("%Y-%m-%d") if last_act else None,
                    "data_source": self.rng.choice(["MDE", "OOD"]),
                    "sale_amt_usd": round(base_usd, 2),
                    "sellin_price": round(base_usd * 0.65, 2),
                    "premium_flg": int(self.rng.random() < 0.35),
                })

        return pd.DataFrame(rows)

    def _generate_app_usage(self, demo: pd.DataFrame) -> pd.DataFrame:
        rows = []
        all_apps = SAMSUNG_APPS + NON_SAMSUNG_APPS
        base_week = pd.Timestamp("2025-01-06")

        for _, cust in demo.iterrows():
            n_apps = int(self.rng.integers(3, 12))
            selected = [all_apps[i] for i in self.rng.choice(len(all_apps), n_apps, replace=False)]
            weeks = int(self.rng.integers(2, 5))

            for app_id, app_title, app_cat in selected:
                is_samsung = any(app_id == a[0] for a in SAMSUNG_APPS)
                for w in range(weeks):
                    wk_start = base_week + pd.Timedelta(weeks=w)
                    wk_end = wk_start + pd.Timedelta(days=6)
                    usage_sec = max(60, int(
                        self.rng.lognormal(8, 1.5) if app_cat == "GAME" else self.rng.lognormal(7, 1.2)
                    ))
                    rows.append({
                        "index": cust["index"],
                        "usage_month": f"2025-W{4 + w:02d}",
                        "app_id": app_id,
                        "app_title": app_title,
                        "app_is_samsung": is_samsung,
                        "app_category": app_cat,
                        "app_game_category": self.rng.choice(["CASUAL", "PUZZLE", "STRATEGY", "RPG"])
                        if app_cat == "GAME" else None,
                        "usage_cnt": int(self.rng.integers(5, 120)),
                        "usage_duration_seconds": usage_sec,
                        "fst_usage_date": wk_start.strftime("%Y-%m-%d"),
                        "lst_usage_date": wk_end.strftime("%Y-%m-%d"),
                        "dvc_cnt": 1,
                        "dvc_model_list": "['SM-S926B']",
                        "usage_month_last_day": wk_end.strftime("%Y-%m-%d"),
                        "cii_load_dt": (wk_end + pd.Timedelta(days=7)).strftime("%Y-%m-%d 03:00:00"),
                    })

        return pd.DataFrame(rows)

    def _generate_interests(self, demo: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, cust in demo.iterrows():
            active = set(self.rng.choice(INTEREST_CATEGORIES, int(self.rng.integers(3, 8)), replace=False))
            for interest in INTEREST_CATEGORIES:
                score = float(self.rng.exponential(0.05) if interest in active else self.rng.uniform(0, 0.005))
                rows.append({
                    "index": cust["index"],
                    "usr_cnty_cd": CNTY_CD_BY_AP2.get(cust["usr_cnty_ap2"], cust["usr_cnty_ap2"]),
                    "interest": interest,
                    "category": interest.replace(" & ", "_AND_").replace(" ", "_"),
                    "SUB_SCORE": round(score, 6),
                    "INTEREST_SCORE": round(score, 6),
                })
        return pd.DataFrame(rows)

    def _generate_rewards(self, demo: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, cust in demo.iterrows():
            if self.rng.random() > 0.60:
                continue

            ap2 = cust["usr_cnty_ap2"]
            currency, fx = CURRENCIES_BY_AP2.get(ap2, ("USD", 1.0))
            tier = self.rng.choice(["Bronze", "Silver", "Gold", "Platinum"], p=[0.40, 0.30, 0.20, 0.10])
            tier_map = {"Bronze": ("T001", 1), "Silver": ("T002", 2), "Gold": ("T003", 3), "Platinum": ("T004", 4)}
            tier_code, tier_level = tier_map[tier]

            acc = int(self.rng.lognormal(9, 1.2))
            rdm = int(acc * self.rng.uniform(0.3, 0.95))
            exp = int(acc * self.rng.uniform(0, 0.05))
            amt = max(0, acc - rdm - exp)

            last_acc = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(self.rng.integers(0, 450)))
            last_rdm = last_acc - pd.Timedelta(days=int(self.rng.integers(0, 90)))
            tier_start = last_acc - pd.Timedelta(days=365)
            tier_end = last_acc + pd.Timedelta(days=365)

            rows.append({
                "index": cust["index"],
                "rewards_cnt": 1,
                "rewards_priority": 1,
                "rwd_cnty_cd_2": CNTY_CD2_BY_AP2.get(ap2, "US"),
                "rwd_cnty_cd": CNTY_CD_BY_AP2.get(ap2, "USA"),
                "rwd_cnty_name": cust["usr_cnty_name"],
                "rwd_cnty_gc": GC_BY_AP2.get(ap2, "N.America"),
                "rwd_cnty_ap2": ap2,
                "point_type": "Samsung Rewards",
                "point_amt": amt,
                "point_acc_amt": acc,
                "point_rdm_amt": -rdm,
                "point_exp_amt": exp,
                "exp_amount_1month": 0,
                "exp_amount_3month": 0,
                "last_date_acc": last_acc.strftime("%Y-%m-%d"),
                "last_date_rdm": last_rdm.strftime("%Y-%m-%d"),
                "tier_level_code": tier_code,
                "tier_name": tier,
                "tier_level": tier_level,
                "tier_start_date": tier_start.strftime("%Y-%m-%d"),
                "tier_end_date": tier_end.strftime("%Y-%m-%d"),
                "best_tier_level_code": tier_code,
                "best_tier_name": tier,
                "best_tier_level": tier_level,
                "best_tier_end_date": tier_end.strftime("%Y-%m-%d"),
                "currency": currency,
                "currency_rate": fx,
                "point_rate": 0.01,
                "point_amt_usd": round(amt * 0.01, 2),
                "point_acc_amt_usd": round(acc * 0.01, 2),
                "point_rdm_amt_usd": round(-rdm * 0.01, 2),
            })

        return pd.DataFrame(rows)

    def load_real_distributions(self, excel_path: str) -> Dict[str, pd.DataFrame]:
        """Extract distribution parameters from real data."""
        return {
            "demo": pd.read_excel(excel_path, sheet_name="Demo", header=1),
            "clv": pd.read_excel(excel_path, sheet_name="CLV", header=1),
            "purchase": pd.read_excel(excel_path, sheet_name="구매", header=1),
            "interests": pd.read_excel(excel_path, sheet_name="관심사", header=1),
            "app_usage": pd.read_excel(excel_path, sheet_name="앱사용", header=1),
        }

    def generate_synthetic_data(self, n: int, dist: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate synthetic customer data based on real distributions."""
        print(f"Generating synthetic data... (n={n})")

        demo = self._gen_synthetic_demo(n, dist["demo"])
        clv = self._gen_synthetic_clv_features(n, dist["clv"])
        app = self._gen_synthetic_app_features(n, dist["app_usage"])
        interest = self._gen_synthetic_interest_features(n, dist["interests"])
        home = self._gen_synthetic_home_appliance_features(n)

        return pd.concat([demo, clv, app, interest, home], axis=1)

    def _gen_synthetic_demo(self, n: int, demo_df: pd.DataFrame) -> pd.DataFrame:
        gender_dist = demo_df["usr_gndr"].value_counts(normalize=True)
        activeness_dist = demo_df["sa_activeness"].value_counts(normalize=True)
        country_dist = demo_df["usr_cnty_ap2"].value_counts(normalize=True)

        age_mean = demo_df["usr_age"].mean()
        age_std = demo_df["usr_age"].std()

        return pd.DataFrame({
            "index": range(1, n + 1),
            "usr_age": np.clip(self.rng.normal(age_mean, age_std, n).astype(int), 18, 75),
            "usr_gndr": self.rng.choice(gender_dist.index, n, p=gender_dist.values),
            "sa_activeness": self.rng.choice(activeness_dist.index, n, p=activeness_dist.values),
            "usr_cnty_ap2": self.rng.choice(country_dist.index, n, p=country_dist.values),
        })

    def _gen_synthetic_clv_features(self, n: int, clv_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "retention_score": np.clip(self.rng.normal(
                clv_df["retention_score"].mean(),
                clv_df["retention_score"].std(), n), 0, 1),
            "ltv_r": np.clip(self.rng.lognormal(
                np.log(clv_df["ltv_r"].clip(lower=1).mean()),
                clv_df["ltv_r"].clip(lower=1).std() / clv_df["ltv_r"].clip(lower=1).mean(), n), 0, None),
            "val_p": np.clip(self.rng.lognormal(
                np.log(clv_df["val_p"].clip(lower=1).mean()),
                0.8, n), 0, None),
            "pchs_cnt": self.rng.integers(1, 10, n),
            "premium_cnt": self.rng.integers(0, 5, n),
            "cum_repchs_flg": self.rng.choice([0, 1], n, p=[
                1 - clv_df["cum_repchs_flg"].mean(),
                clv_df["cum_repchs_flg"].mean()]),
            "samsunghealth_flag": self.rng.choice([0, 1], n, p=[
                1 - clv_df["samsunghealth_flag"].mean(),
                clv_df["samsunghealth_flag"].mean()]),
            "samsungwallet_flag": self.rng.choice([0, 1], n, p=[
                1 - clv_df["samsungwallet_flag"].mean(),
                clv_df["samsungwallet_flag"].mean()]),
            "smartthings_flag": self.rng.choice([0, 1], n, p=[
                1 - clv_df["smartthings_flag"].mean(),
                clv_df["smartthings_flag"].mean()]),
        })

    def _gen_synthetic_home_appliance_features(self, n: int) -> pd.DataFrame:
        household_types = ["Single", "Two-person", "Multi-person", "With Children"]
        lifestyle_types = ["Practical", "Minimalist", "Smart Home Enthusiast", "Design Focused", "Tech Savvy"]
        promotion_prefs = ["Bundle Discount", "Trade-in", "Extended Warranty", "Reward Points"]
        loyalty_types = ["Loyal", "Switcher", "First-time Buyer", "Competitor User"]
        decision_factors = ["Performance", "Design", "Price/Value", "Brand Trust", "Ecosystem/Connectivity"]
        info_channels = ["YouTube/SNS", "Official Website", "Offline Store", "Friend/Family Review"]
        previous_brands = ["Samsung", "LG", "Apple", "Sony", "Dyson", "None(First)"]
        rewards_tiers = ["Bronze", "Silver", "Gold", "Platinum"]

        return pd.DataFrame({
            "[option]household_type": self.rng.choice(household_types, n, p=[0.3, 0.3, 0.2, 0.2]),
            "[option]pet_owned": self.rng.choice([0, 1], n, p=[0.7, 0.3]),
            "[option]lifestyle_type": self.rng.choice(lifestyle_types, n),
            "[option]home_ownership": self.rng.choice(["Own", "Rent"], n, p=[0.6, 0.4]),
            "[option]own_refrigerator": self.rng.choice([0, 1], n, p=[0.6, 0.4]),
            "[option]own_washer": self.rng.choice([0, 1], n, p=[0.7, 0.3]),
            "[option]own_dryer": self.rng.choice([0, 1], n, p=[0.8, 0.2]),
            "[option]own_airconditioner": self.rng.choice([0, 1], n, p=[0.5, 0.5]),
            "[option]own_tv": self.rng.choice([0, 1], n, p=[0.4, 0.6]),
            "[option]smartthings_usage_level": np.clip(self.rng.normal(30, 25, n), 0, 100).astype(int),
            "[option]home_ecosystem_score": np.clip(self.rng.normal(40, 30, n), 0, 100).astype(int),
            "[option]loyalty_type": self.rng.choice(loyalty_types, n, p=[0.4, 0.3, 0.1, 0.2]),
            "[option]previous_brand": self.rng.choice(previous_brands, n),
            "[option]purchase_decision_factor": self.rng.choice(decision_factors, n),
            "[option]preferred_info_channel": self.rng.choice(info_channels, n),
            "[option]preferred_promotion": self.rng.choice(promotion_prefs, n),
            "[option]rewards_tier": self.rng.choice(rewards_tiers, n, p=[0.5, 0.3, 0.15, 0.05]),
            "[option]satisfaction_sentiment": np.clip(self.rng.normal(70, 20, n), 0, 100).astype(int),
        })

    def _gen_synthetic_app_features(self, n: int, app_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "app_game_ratio": self.rng.beta(2, 5, n),
            "app_social_ratio": self.rng.beta(3, 4, n),
            "app_samsung_ratio": np.clip(self.rng.normal(
                app_df["app_is_samsung"].mean(), 0.1, n), 0, 1),
            "avg_daily_usage_min": np.clip(self.rng.normal(120, 60, n), 10, 600),
        })

    def _gen_synthetic_interest_features(self, n: int, interest_df: pd.DataFrame) -> pd.DataFrame:
        top_interests = interest_df.groupby("interest")["INTEREST_SCORE"].mean().nlargest(8)

        features = {}
        for interest in top_interests.index:
            col = f"interest_{interest.lower().replace(' & ', '_').replace(' ', '_')}"
            score_mean = interest_df[interest_df["interest"] == interest]["INTEREST_SCORE"].mean()
            features[col] = np.clip(self.rng.exponential(score_mean, n), 0, 1)

        return pd.DataFrame(features)
