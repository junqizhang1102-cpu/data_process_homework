"""
LLM Forecast Module for CO2 Analysis
Extracted from llm_forecast.py notebook
"""

import os
import json
import numpy as np
import pandas as pd
import re
import csv
from datetime import datetime, timedelta
import requests
import time
import math

def extract_float(text):
    """Extract float number from text response"""
    match = re.search(r'[-+]?\d+(?:\.\d+)?', text)
    return float(match.group(0)) if match else None

def load_cache(cache_file):
    """Load cache from JSON file"""
    if os.path.isfile(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache, cache_file):
    """Save cache to JSON file"""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)

def save_csv(data, filename, header=None):
    """Save data to CSV file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(data)

def save_json(data, filename):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def mae(preds, trues):
    """Mean Absolute Error"""
    errors = np.abs(np.array(preds) - np.array(trues))
    return float(np.mean(errors))

def rmse(preds, trues):
    """Root Mean Square Error"""
    errors = np.array(preds) - np.array(trues)
    return float(np.sqrt(np.mean(errors**2)))

def mape(preds, trues):
    """Mean Absolute Percentage Error"""
    errors = np.abs((np.array(preds) - np.array(trues)) / np.array(trues))
    # Filter out infinite values and zeros in denominator
    valid_mask = np.isfinite(errors) & (trues != 0)
    return float(np.mean(errors[valid_mask])) if np.any(valid_mask) else float('nan')

def forecast_llm_fair_comparison(df_co2, fit_rows, context_weeks=12, api_key=None, model='gpt-4o', base_url=None, cache=None, cache_key_prefix=''):
    """
    Generate LLM forecasts for fair comparison with existing methods.
    Uses the same train/test split as other models (BAU, Ridge, LassoCV, GBR).
    
    Args:
        df_co2: DataFrame with CO2 data containing 'time' and 'CO2_ppm' columns
        fit_rows: Boolean mask indicating training period (same as used for other models)
        context_weeks: Number of weeks to use as context for LLM forecasting
        api_key: OpenAI API key
        model: LLM model to use
        base_url: Base URL for API
        cache: Cache dictionary for LLM responses
        cache_key_prefix: Prefix for cache keys
    
    Returns:
        dict: Contains 'train_pred', 'test_pred', 'train_r2', 'test_r2', 'train_rmse', 'test_rmse', 'test_indices'
    """
    if cache is None:
        cache = {}
    
    # Get train/test split points
    train_end_idx = np.where(~fit_rows)[0][0] if np.any(~fit_rows) else len(df_co2)
    test_start_idx = train_end_idx
    
    # Initialize arrays for predictions that match the original dataframe length
    llm_ppm = np.full(len(df_co2), np.nan)
    
    # Get training and test data (keep original indices)
    train_data = df_co2.iloc[:train_end_idx].copy()
    test_data = df_co2.iloc[test_start_idx:].copy()
    
    # Store original test indices
    test_indices = test_data.index.tolist()
    
    # Remove any rows with NaN values (but keep track of original indices)
    train_data_clean = train_data.dropna(subset=['CO2_ppm', 'time'])
    test_data_clean = test_data.dropna(subset=['CO2_ppm', 'time'])
    
    # Generate predictions for training period (to get training metrics)
    train_predictions = []
    train_actuals = []
    
    print(f"\n正在生成训练期LLM预测（从第{context_weeks}个点开始）...")
    for i in range(context_weeks, len(train_data_clean)):
        # Get context - 只使用历史真实数据
        context_start = max(0, i - context_weeks)
        context_data = train_data_clean.iloc[context_start:i]
        
        # Skip if not enough context data
        if len(context_data) == 0:
            continue
            
        # Prepare context for LLM
        context_text = ""
        for _, row in context_data.iterrows():
            context_text += f"Time {row['time']:.4f}: {row['CO2_ppm']:.2f} ppm\n"
        
        # Get current info
        current_time = train_data_clean.iloc[i]['time']
        current_co2 = train_data_clean.iloc[i]['CO2_ppm']
        
        # 使用真正的LLM API调用
        prediction = call_llm_api(context_text, current_time, api_key, model, base_url, cache)
        
        # 如果API调用失败，使用趋势外推作为fallback
        if prediction is None:
            if len(context_data) > 1:
                from scipy import stats
                times = context_data['time'].values
                values = context_data['CO2_ppm'].values
                slope, intercept, _, _, _ = stats.linregress(times, values)
                prediction = slope * current_time + intercept
            elif len(context_data) == 1:
                prediction = context_data['CO2_ppm'].iloc[-1]
            else:
                prediction = current_co2
        
        train_predictions.append(prediction)
        train_actuals.append(current_co2)
        
        if (i + 1) % 50 == 0:
            print(f"  训练期: 已完成 {i+1}/{len(train_data_clean)} 个预测")
    
    # Generate predictions for test period
    test_predictions = []
    test_actuals = []
    
    print(f"\n正在生成测试期LLM预测...")
    # 将训练数据和测试数据合并，但只在预测时使用已知的历史数据
    all_data_so_far = train_data_clean.copy()
    
    for i, (test_idx, test_row) in enumerate(test_data_clean.iterrows()):
        current_time = test_row['time']
        current_co2 = test_row['CO2_ppm']
        
        # 只使用到当前时间点之前的所有真实数据作为上下文（训练数据 + 之前的测试数据）
        # 注意：不使用当前点的真实值，避免数据泄露
        context_data = all_data_so_far.tail(context_weeks)
        
        # Skip if not enough context
        if len(context_data) == 0:
            # 使用训练数据的最后几个点
            context_data = train_data_clean.tail(context_weeks)
        
        # Prepare context for LLM
        context_text = ""
        for _, row in context_data.iterrows():
            context_text += f"Time {row['time']:.4f}: {row['CO2_ppm']:.2f} ppm\n"
        
        # 使用真正的LLM API调用
        prediction = call_llm_api(context_text, current_time, api_key, model, base_url, cache)
        
        # 如果API调用失败，使用趋势外推作为fallback
        if prediction is None:
            if len(context_data) > 1:
                from scipy import stats
                times = context_data['time'].values
                values = context_data['CO2_ppm'].values
                slope, intercept, _, _, _ = stats.linregress(times, values)
                prediction = slope * current_time + intercept
            elif len(context_data) == 1:
                prediction = context_data['CO2_ppm'].iloc[-1]
            else:
                prediction = train_data_clean['CO2_ppm'].iloc[-1]
        
        # Store prediction in the original dataframe position
        llm_ppm[test_idx] = prediction
        
        test_predictions.append(prediction)
        test_actuals.append(current_co2)
        
        # 预测完成后，将当前的真实数据添加到历史中，用于下一个预测
        all_data_so_far = pd.concat([all_data_so_far, pd.DataFrame({
            'time': [current_time],
            'CO2_ppm': [current_co2]
        })], ignore_index=True)
        
        if (i + 1) % 20 == 0:
            print(f"  测试期: 已完成 {i+1}/{len(test_data_clean)} 个预测")
    
    # Calculate metrics
    from sklearn.metrics import r2_score
    
    # Ensure we have enough data points for metrics calculation
    if len(train_actuals) < 2 or len(train_predictions) < 2:
        train_r2 = 0.0
        train_rmse = 0.0
    else:
        train_r2 = r2_score(train_actuals, train_predictions)
        train_rmse = rmse(train_predictions, train_actuals)
    
    if len(test_actuals) < 2 or len(test_predictions) < 2:
        test_r2 = 0.0
        test_rmse = 0.0
    else:
        test_r2 = r2_score(test_actuals, test_predictions)
        test_rmse = rmse(test_predictions, test_actuals)
    
    return {
        'train_pred': train_predictions,
        'test_pred': test_predictions,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'llm_ppm': llm_ppm  # This is the array that matches the original dataframe length
    }

def call_llm_api(context_text, target_time, api_key, model='gpt-4o', base_url=None, cache=None):
    """
    Call LLM API to predict CO2 for a specific time point
    
    Args:
        context_text: Context data formatted as text
        target_time: Target time to predict
        api_key: API key
        model: Model name
        base_url: API base URL
        cache: Cache dictionary
    
    Returns:
        float: Predicted CO2 value
    """
    # Create cache key
    cache_key = f"{model}_{target_time}_{hash(context_text)}"
    
    # Check cache
    if cache and cache_key in cache:
        return cache[cache_key]
    
    # Prepare prompt
    prompt = f"""Based on the following weekly CO2 concentration data (in ppm), predict the CO2 concentration for time {target_time:.4f}.

Historical data:
{context_text}

Please analyze the trend and seasonal patterns, then provide ONLY a single number as your prediction for time {target_time:.4f}.
Your answer should be just the predicted CO2 value in ppm (e.g., 425.50), nothing else."""

    try:
        # Call API
        if base_url is None:
            base_url = 'https://api.openai.com/v1'
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
        data = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': 'You are a time series forecasting expert specializing in CO2 concentration prediction.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.1,
            'max_tokens': 50
        }
        
        response = requests.post(f'{base_url}/chat/completions', headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        prediction_text = result['choices'][0]['message']['content'].strip()
        
        # Extract number from response
        prediction = extract_float(prediction_text)
        
        if prediction is None:
            raise ValueError(f"Could not extract number from: {prediction_text}")
        
        # Cache result
        if cache is not None:
            cache[cache_key] = prediction
        
        return prediction
        
    except Exception as e:
        print(f"Warning: LLM API call failed for time {target_time}: {e}")
        return None


def predict_future_with_llm(df_co2, context_weeks=12, api_key=None, model='gpt-4o', base_url=None, cache=None):
    """
    Use LLM to predict future CO2 values for points without observations
    
    注意：此函数用于预测未来（没有真实观测值的时期）。
    策略：使用最后的真实观测数据作为上下文，不使用预测值作为上下文，
    因为预测值的误差会累积。这与其他统计模型（Ridge/Lasso/GBR）一致，
    它们也基于历史真实数据的拟合来外推未来。
    
    Args:
        df_co2: DataFrame with CO2 data
        context_weeks: Number of weeks to use as context (建议使用更多历史数据，如52周=1年)
        api_key: API key
        model: Model name
        base_url: API base URL
        cache: Cache dictionary
    
    Returns:
        dict: Dictionary mapping index to predicted value
    """
    predictions = {}
    
    # Find future points (no actual CO2_ppm data)
    future_mask = df_co2['CO2_ppm'].isna()
    future_indices = df_co2[future_mask].index.tolist()
    
    if len(future_indices) == 0:
        return predictions
    
    # Get last valid observation index
    last_valid_idx = df_co2['CO2_ppm'].last_valid_index()
    
    if last_valid_idx is None:
        return predictions
    
    # Get context data - 使用更多历史数据以提高预测质量
    # 使用至少52周（1年）或context_weeks，取较大值
    actual_context_weeks = max(context_weeks, 52)
    context_start_idx = max(0, last_valid_idx - actual_context_weeks + 1)
    context_data = df_co2.loc[context_start_idx:last_valid_idx, ['time', 'CO2_ppm']].dropna()
    
    if len(context_data) < 2:
        return predictions
    
    print(f"\n{'='*60}")
    print(f"使用LLM预测 {len(future_indices)} 个未来数据点")
    print(f"上下文窗口: {len(context_data)} 周（最后观测: {context_data['time'].iloc[-1]:.4f}）")
    print(f"预测策略: 基于历史真实数据，不使用预测值避免误差累积")
    print(f"{'='*60}")
    
    # Predict each future point independently, all using the same historical context
    # 不使用预测值作为上下文，避免误差累积
    for i, future_idx in enumerate(future_indices):
        future_time = df_co2.loc[future_idx, 'time']
        
        # 始终使用相同的历史真实数据作为上下文
        context_text = ""
        for _, row in context_data.tail(actual_context_weeks).iterrows():
            context_text += f"Time {row['time']:.4f}: {row['CO2_ppm']:.2f} ppm\n"
        
        # Call LLM
        prediction = call_llm_api(context_text, future_time, api_key, model, base_url, cache)
        
        if prediction is not None:
            predictions[future_idx] = prediction
            
            if (i + 1) % 10 == 0:
                print(f"  已完成 {i + 1}/{len(future_indices)} 个预测")
        else:
            print(f"  警告: 时间点 {future_time:.4f} 预测失败，使用趋势外推")
            # Fallback to trend extrapolation using historical data
            if len(context_data) >= 2:
                times = context_data['time'].tail(actual_context_weeks).values
                values = context_data['CO2_ppm'].tail(actual_context_weeks).values
                from scipy import stats
                slope, intercept, _, _, _ = stats.linregress(times, values)
                prediction = slope * future_time + intercept
                predictions[future_idx] = prediction
    
    print(f"✓ 完成所有 {len(predictions)} 个未来点的LLM预测")
    print(f"{'='*60}\n")
    return predictions


# Alias for backward compatibility
forecast_llm = forecast_llm_fair_comparison