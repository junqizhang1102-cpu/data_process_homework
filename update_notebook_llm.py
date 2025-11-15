import json
import os

NB_PATH = os.path.join(os.getcwd(), "notebook303e42d520.ipynb")

cell1 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "import os,json\n",
        "from llm_forecast import forecast_llm, mae, rmse, mape, load_cache, save_cache, save_csv, save_json\n",
        "MODEL = 'gpt-4o'\n",
        "BASE_URL = os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1'\n",
        "API_KEY = os.environ.get('OPENAI_API_KEY')\n",
        "CONTEXT_WEEKS = 12\n",
        "EVAL_WEEKS = 52\n",
        "OUT_DIR = 'results_notebook'\n",
        "CACHE_FILE = os.path.join(OUT_DIR, 'llm_cache.json')\n",
        "os.makedirs(OUT_DIR, exist_ok=True)\n",
        "cache = load_cache(CACHE_FILE) if os.path.isfile(CACHE_FILE) else {}\n",
        "data = [(float(f\"{t:.6f}\"), float(v)) for t, v in zip(df_co2['time'].values.tolist(), df_co2['CO2_ppm'].values.tolist()) if v >= 0]\n",
        "start_index = max(0, len(data) - EVAL_WEEKS)\n",
        "preds, trues, decs = forecast_llm(data, start_index, EVAL_WEEKS, CONTEXT_WEEKS, API_KEY, MODEL, BASE_URL, None, False, cache, 'notebook')\n",
        "save_cache(cache, CACHE_FILE)\n",
    ],
}

cell2 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "dec_to_bau = {float(f\"{t:.6f}\"): float(v) for t, v in zip(df_co2['time'].values.tolist(), df_co2['BAU_ppm'].values.tolist())}\n",
        "bau_vals = [dec_to_bau[d] for d in decs]\n",
        "llm_metrics = {\"mae\": mae(preds, trues), \"rmse\": rmse(preds, trues), \"mape\": mape(preds, trues), \"n\": len(preds)}\n",
        "bau_metrics = {\"mae\": mae(bau_vals, trues), \"rmse\": rmse(bau_vals, trues), \"mape\": mape(bau_vals, trues), \"n\": len(bau_vals)}\n",
        "print(json.dumps({\"llm\": llm_metrics, \"bau\": bau_metrics}, ensure_ascii=False))\n",
        "save_json(llm_metrics, os.path.join(OUT_DIR, \"metrics_llm_all.json\"))\n",
        "save_json(bau_metrics, os.path.join(OUT_DIR, \"metrics_bau_aligned.json\"))\n",
        "rows = [[d, y, p, b] for d, y, p, b in zip(decs, trues, preds, bau_vals)]\n",
        "save_csv(rows, os.path.join(OUT_DIR, \"eval_aligned.csv\"), header=[\"decimal\",\"true\",\"llm_pred\",\"bau_pred\"])\n",
    ],
}

cell3 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "import matplotlib.pyplot as plt\n",
        "plt.figure(figsize=(10,5))\n",
        "plt.plot(decs, trues, label='truth')\n",
        "plt.plot(decs, preds, label='llm')\n",
        "plt.plot(decs, bau_vals, label='bau')\n",
        "plt.legend()\n",
        "plt.title(\"Aligned: LLM vs BAU vs Truth\")\n",
        "plt.xlabel(\"decimal_year\")\n",
        "plt.ylabel(\"ppm\")\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(OUT_DIR, \"plot_aligned_nb.png\"))\n",
        "plt.show()\n",
    ],
}

def has_llm_cells(nb):
    for c in nb.get("cells", []):
        src = "".join(c.get("source", []))
        if "forecast_llm(" in src and "llm_pred" in src:
            return True
    return False

def has_llm_long(nb):
    for c in nb.get("cells", []):
        src = "".join(c.get("source", []))
        if "LLM_LONG" in src:
            return True
    return False

def find_cell_index(nb, keyword):
    """Return first cell index containing keyword or -1"""
    for idx, c in enumerate(nb.get("cells", [])):
        if keyword in "".join(c.get("source", [])):
            return idx
    return -1

def main():
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
    if not has_llm_cells(nb):
        nb.setdefault("cells", []).extend([cell1, cell2, cell3])
    if not has_llm_long(nb):
        nb.setdefault("cells", []).extend([
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"trusted": True},
                "outputs": [],
                "source": [
                    "# Build additional ML methods (GBR/Lasso/Ridge) on same features and fit-range as BAU\n",
                    "import numpy as np\n",
                    "from sklearn.ensemble import GradientBoostingRegressor\n",
                    "from sklearn.linear_model import Lasso, Ridge\n",
                    "from sklearn.preprocessing import StandardScaler\n",
                    "yr_fit_val = 2019.0 + (74.0-0.5)/365.0\n",
                    "times = df_co2['time'].values\n",
                    "vals = df_co2['CO2_ppm'].values\n",
                    "phases = times - np.floor(times)\n",
                    "time1 = times - yr_fit_val\n",
                    "time2 = (times - yr_fit_val)**2\n",
                    "feats = [time1, time2]\n",
                    "for ih in range(1,6):\n",
                    "    feats.append(np.sin(2.0*np.pi*ih*phases))\n",
                    "    feats.append(np.cos(2.0*np.pi*ih*phases))\n",
                    "X = np.vstack(feats).T\n",
                    "fit_rows = times < yr_fit_val\n",
                    "methods_maps = {}\n",
                    "# GBR\n",
                    "try:\n",
                    "    gbr = GradientBoostingRegressor(random_state=42)\n",
                    "    gbr.fit(X[fit_rows], vals[fit_rows])\n",
                    "    gbr_pred = gbr.predict(X)\n",
                    "    df_co2['GBR_ppm'] = gbr_pred\n",
                    "    methods_maps['gbr'] = {float(f\"{t:.6f}\"): float(p) for t,p in zip(times, gbr_pred)}\n",
                    "except Exception:\n",
                    "    pass\n",
                    "# Lasso (scaled)\n",
                    "try:\n",
                    "    sc = StandardScaler()\n",
                    "    Xs = sc.fit_transform(X[fit_rows])\n",
                    "    ls = Lasso(alpha=0.01, random_state=42) if 'random_state' in Lasso().get_params() else Lasso(alpha=0.01)\n",
                    "    ls.fit(Xs, vals[fit_rows])\n",
                    "    lasso_pred = ls.predict(sc.transform(X))\n",
                    "    df_co2['Lasso_ppm'] = lasso_pred\n",
                    "    methods_maps['lasso'] = {float(f\"{t:.6f}\"): float(p) for t,p in zip(times, lasso_pred)}\n",
                    "except Exception:\n",
                    "    pass\n",
                    "# Ridge (scaled)\n",
                    "try:\n",
                    "    sc2 = StandardScaler()\n",
                    "    Xs2 = sc2.fit_transform(X[fit_rows])\n",
                    "    rg = Ridge(alpha=1.0, random_state=42) if 'random_state' in Ridge().get_params() else Ridge(alpha=1.0)\n",
                    "    rg.fit(Xs2, vals[fit_rows])\n",
                    "    ridge_pred = rg.predict(sc2.transform(X))\n",
                    "    df_co2['Ridge_ppm'] = ridge_pred\n",
                    "    methods_maps['ridge'] = {float(f\"{t:.6f}\"): float(p) for t,p in zip(times, ridge_pred)}\n",
                    "except Exception:\n",
                    "    pass\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"trusted": True},
                "outputs": [],
                "source": [
                    "LLM_LONG = True\n",
                    "import os,json\n",
                    "from llm_forecast import forecast_llm, mae, rmse, mape, load_cache, save_cache, save_csv, save_json\n",
                    "MODEL = 'gpt-4o'\n",
                    "BASE_URL = os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1'\n",
                    "API_KEY = os.environ.get('OPENAI_API_KEY')\n",
                    "CONTEXT_WEEKS = 12\n",
                    "OUT_DIR = 'results_notebook'\n",
                    "CACHE_FILE = os.path.join(OUT_DIR, 'llm_cache.json')\n",
                    "MAX_FORECASTS = None\n",
                    "os.makedirs(OUT_DIR, exist_ok=True)\n",
                    "cache = load_cache(CACHE_FILE) if os.path.isfile(CACHE_FILE) else {}\n",
                    "data = [(float(f\"{t:.6f}\"), float(v)) for t, v in zip(df_co2['time'].values.tolist(), df_co2['CO2_ppm'].values.tolist()) if v >= 0]\n",
                    "yr_fit_val = float(df_co2.loc[df_co2['time'] < 3000,'time'][0])\n",
                    "yr_fit_val = 2019.0 + (74.0-0.5)/365.0\n",
                    "start_index = 0\n",
                    "for i,(t,_) in enumerate(data):\n",
                    "    if t >= yr_fit_val:\n",
                    "        start_index = i\n",
                    "        break\n",
                    "eval_weeks = len(data) - start_index\n",
                    "preds_long, trues_long, decs_long = forecast_llm(data, start_index, eval_weeks, CONTEXT_WEEKS, API_KEY, MODEL, BASE_URL, MAX_FORECASTS, False, cache, 'notebook_long')\n",
                    "save_cache(cache, CACHE_FILE)\n",
                    "rows = [[d, y, p] for d, y, p in zip(decs_long, trues_long, preds_long)]\n",
                    "save_csv(rows, os.path.join(OUT_DIR, 'llm_long_predictions.csv'), header=['decimal','true','llm_pred'])\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"trusted": True},
                "outputs": [],
                "source": [
                    "dec_to_bau = {float(f\"{t:.6f}\"): float(v) for t, v in zip(df_co2['time'].values.tolist(), df_co2['BAU_ppm'].values.tolist())}\n",
                    "trues_long_aligned = [y for d,y in zip(decs_long, trues_long) if d in dec_to_bau]\n",
                    "llm_vals_long_aligned = [p for d,p in zip(decs_long, preds_long) if d in dec_to_bau]\n",
                    "bau_vals_long_aligned = [dec_to_bau[d] for d in decs_long if d in dec_to_bau]\n",
                    "llm_metrics_long = {'mae': mae(llm_vals_long_aligned, trues_long_aligned), 'rmse': rmse(llm_vals_long_aligned, trues_long_aligned), 'mape': mape(llm_vals_long_aligned, trues_long_aligned), 'n': len(trues_long_aligned)}\n",
                    "bau_metrics_long = {'mae': mae(bau_vals_long_aligned, trues_long_aligned), 'rmse': rmse(bau_vals_long_aligned, trues_long_aligned), 'mape': mape(bau_vals_long_aligned, trues_long_aligned), 'n': len(trues_long_aligned)}\n",
                    "import pandas as pd\n",
                    "table_all = pd.DataFrame([\n",
                    "    {'method':'LLM','mae': llm_metrics_long['mae'], 'rmse': llm_metrics_long['rmse'], 'mape': llm_metrics_long['mape'], 'n': llm_metrics_long['n']},\n",
                    "    {'method':'BAU','mae': bau_metrics_long['mae'], 'rmse': bau_metrics_long['rmse'], 'mape': bau_metrics_long['mape'], 'n': bau_metrics_long['n']},\n",
                    "])\n",
                    "print(table_all)\n",
                    "table_all.to_csv(os.path.join(OUT_DIR, 'metrics_methods_table_long_all.csv'), index=False)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"trusted": True},
                "outputs": [],
                "source": [
                    "sr15_map = {float(f\"{t:.6f}\"): float(v) for t, v in zip(df_co2['time'].values.tolist(), df_co2['SR15_ppm'].values.tolist()) if pd.notna(v)}\n",
                    "decs_sr = [d for d in decs_long if d in sr15_map and d in dec_to_bau]\n",
                    "trues_sr = [y for d,y in zip(decs_long, trues_long) if d in sr15_map and d in dec_to_bau]\n",
                    "llm_sr = [p for d,p in zip(decs_long, preds_long) if d in sr15_map and d in dec_to_bau]\n",
                    "bau_sr = [dec_to_bau[d] for d in decs_sr]\n",
                    "sr15_vals = [sr15_map[d] for d in decs_sr]\n",
                    "llm_metrics_sr = {'mae': mae(llm_sr, trues_sr), 'rmse': rmse(llm_sr, trues_sr), 'mape': mape(llm_sr, trues_sr), 'n': len(trues_sr)}\n",
                    "bau_metrics_sr = {'mae': mae(bau_sr, trues_sr), 'rmse': rmse(bau_sr, trues_sr), 'mape': mape(bau_sr, trues_sr), 'n': len(trues_sr)}\n",
                    "sr15_metrics_sr = {'mae': mae(sr15_vals, trues_sr), 'rmse': rmse(sr15_vals, trues_sr), 'mape': mape(sr15_vals, trues_sr), 'n': len(trues_sr)}\n",
                    "table_sr = pd.DataFrame([\n",
                    "    {'method':'LLM','mae': llm_metrics_sr['mae'], 'rmse': llm_metrics_sr['rmse'], 'mape': llm_metrics_sr['mape'], 'n': llm_metrics_sr['n']},\n",
                    "    {'method':'BAU','mae': bau_metrics_sr['mae'], 'rmse': bau_metrics_sr['rmse'], 'mape': bau_metrics_sr['mape'], 'n': bau_metrics_sr['n']},\n",
                    "    {'method':'SR15','mae': sr15_metrics_sr['mae'], 'rmse': sr15_metrics_sr['rmse'], 'mape': sr15_metrics_sr['mape'], 'n': sr15_metrics_sr['n']},\n",
                    "])\n",
                    "print(table_sr)\n",
                    "table_sr.to_csv(os.path.join(OUT_DIR, 'metrics_methods_table_sr15_subset.csv'), index=False)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"trusted": True},
                "outputs": [],
                "source": [
                    "import matplotlib.pyplot as plt\n",
                    "plt.figure(figsize=(12,6))\n",
                    "plt.plot(df_co2['time'], df_co2['CO2_ppm'], label='truth', linestyle='None', marker='.', markersize=3)\n",
                    "plt.plot(df_co2['time'], df_co2['BAU_ppm'], label='bau', linewidth=1)\n",
                    "plt.plot(decs_long, preds_long, label='llm', linewidth=1)\n",
                    "try:\n",
                    "    plt.plot(df_co2['time'], df_co2['SR15_ppm'], label='sr15', linewidth=1)\n",
                    "except Exception:\n",
                    "    pass\n",
                    "plt.legend()\n",
                    "plt.title('Long horizon: Truth vs LLM vs BAU')\n",
                    "plt.xlabel('decimal_year')\n",
                    "plt.ylabel('ppm')\n",
                    "plt.tight_layout()\n",
                    "save_csv([[d,y,p,b,sr15_map.get(d,None)] for d,y,p,b in zip(decs_long, trues_long, preds_long, [dec_to_bau.get(d) for d in decs_long])], os.path.join(OUT_DIR, 'eval_long_aligned.csv'), header=['decimal','true','llm_pred','bau_pred','sr15_pred'])\n",
                    "plt.savefig(os.path.join(OUT_DIR, 'plot_llm_bau_sr15_truth_long.png'))\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"trusted": True},
                "outputs": [],
                "source": [
                    "import os,glob,pandas as pd\n",
                    "ML_DIR = 'ml_methods'\n",
                    "if os.path.isdir(ML_DIR):\n",
                    "    for p in glob.glob(os.path.join(ML_DIR,'*.csv')):\n",
                    "        name = os.path.splitext(os.path.basename(p))[0]\n",
                    "        dfm = pd.read_csv(p)\n",
                    "        cols = [c.lower() for c in dfm.columns]\n",
                    "        if 'decimal' in cols and 'pred' in cols:\n",
                    "            dcol = dfm.columns[cols.index('decimal')]\n",
                    "            pcol = dfm.columns[cols.index('pred')]\n",
                    "            mp = {float(f\"{t:.6f}\"): float(v) for t, v in zip(dfm[dcol].values.tolist(), dfm[pcol].values.tolist())}\n",
                    "            methods_maps[name] = mp\n",
                    "all_rows = []\n",
                    "metrics_rows = [\n",
                    "    {'method':'LLM','mae': llm_metrics_long['mae'], 'rmse': llm_metrics_long['rmse'], 'mape': llm_metrics_long['mape'], 'n': llm_metrics_long['n']},\n",
                    "    {'method':'BAU','mae': bau_metrics_long['mae'], 'rmse': bau_metrics_long['rmse'], 'mape': bau_metrics_long['mape'], 'n': bau_metrics_long['n']},\n",
                    "    *[{'method': mname, 'mae': mae([methods_maps[mname][d] for d in [dd for dd in decs_long if dd in methods_maps[mname]]], [y for d,y in zip(decs_long, trues_long) if d in methods_maps[mname]]), 'rmse': rmse([methods_maps[mname][d] for d in [dd for dd in decs_long if dd in methods_maps[mname]]], [y for d,y in zip(decs_long, trues_long) if d in methods_maps[mname]]), 'mape': mape([methods_maps[mname][d] for d in [dd for dd in decs_long if dd in methods_maps[mname]]], [y for d,y in zip(decs_long, trues_long) if d in methods_maps[mname]]), 'n': len([y for d,y in zip(decs_long, trues_long) if d in methods_maps[mname]])} for mname in methods_maps]\n",
                    "]\n",
                    "for mname, mmap in methods_maps.items():\n",
                    "    aligned_decs = [d for d in decs_long if d in mmap]\n",
                    "    aligned_trues = [y for d,y in zip(decs_long, trues_long) if d in mmap]\n",
                    "    aligned_preds = [mmap[d] for d in aligned_decs]\n",
                    "    mm = {'mae': mae(aligned_preds, aligned_trues), 'rmse': rmse(aligned_preds, aligned_trues), 'mape': mape(aligned_preds, aligned_trues), 'n': len(aligned_trues)}\n",
                    "    metrics_rows.append({'method': mname, 'mae': mm['mae'], 'rmse': mm['rmse'], 'mape': mm['mape'], 'n': mm['n']})\n",
                    "table_all_methods = pd.DataFrame(metrics_rows)\n",
                    "print(table_all_methods)\n",
                    "table_all_methods.to_csv(os.path.join(OUT_DIR, 'metrics_methods_table_all_long.csv'), index=False)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"trusted": True},
                "outputs": [],
                "source": [
                    "import matplotlib.pyplot as plt\n",
                    "plt.figure(figsize=(12,6))\n",
                    "plt.plot(df_co2['time'], df_co2['CO2_ppm'], label='truth', linestyle='None', marker='.', markersize=3)\n",
                    "plt.plot(df_co2['time'], df_co2['BAU_ppm'], label='bau', linewidth=1)\n",
                    "plt.plot(decs_long, preds_long, label='llm', linewidth=1)\n",
                    "try:\n",
                    "    plt.plot(df_co2['time'], df_co2['SR15_ppm'], label='sr15', linewidth=1)\n",
                    "except Exception:\n",
                    "    pass\n",
                    "for mname, mmap in methods_maps.items():\n",
                    "    xs = [d for d in decs_long if d in mmap]\n",
                    "    ys = [mmap[d] for d in xs]\n",
                    "    plt.plot(xs, ys, label=mname, linewidth=1)\n",
                    "plt.legend(ncol=2)\n",
                    "plt.title('Long horizon: Truth vs LLM vs BAU vs SR15 vs other MLs')\n",
                    "plt.xlabel('decimal_year')\n",
                    "plt.ylabel('ppm')\n",
                    "plt.tight_layout()\n",
                    "plt.savefig(os.path.join(OUT_DIR, 'plot_all_methods_long.png'))\n",
                    "plt.show()\n",
                ],
            },
        ])
        # Reorder cells so LLM prediction cell creating LLM_resid appears before residual scatter plot cell
    try:
        pred_idx = find_cell_index(nb, "LLM_resid")
        if pred_idx == -1:
            pred_idx = find_cell_index(nb, "LLM_ppm")
        resid_plot_idx = find_cell_index(nb, "Residual Comparison by Model")
        if pred_idx != -1 and resid_plot_idx != -1 and pred_idx > resid_plot_idx:
            cell = nb["cells"].pop(pred_idx)
            nb["cells"].insert(resid_plot_idx, cell)
    except Exception:
        pass

    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False)

if __name__ == "__main__":
    main()