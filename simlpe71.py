import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os
from PIL import Image
import io
import base64
from collections import Counter
import numpy as np

# Page config
st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'trades' not in st.session_state:
    st.session_state.trades = []

if 'edit_index' not in st.session_state:
    st.session_state.edit_index = None

if 'start_capital' not in st.session_state:
    st.session_state.start_capital = 1000.0

TRADES_FILE = "trading_journal.json"

# ----- FILE OPERATIONS -----
def save_trades():
    """Save trades to JSON file"""
    trades_to_save = []
    for trade in st.session_state.trades:
        trade_copy = trade.copy()
        if 'images' in trade_copy and trade_copy['images']:
            encoded_images = []
            for img in trade_copy['images']:
                if img is not None:
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    encoded_images.append(base64.b64encode(buf.getvalue()).decode())
            trade_copy['images'] = encoded_images
        else:
            trade_copy['images'] = []
        trades_to_save.append(trade_copy)
    
    with open(TRADES_FILE, 'w', encoding='utf-8') as f:
        json.dump(trades_to_save, f, default=str, indent=2)
    st.toast(f"💾 Saved {len(st.session_state.trades)} trades", icon="✅")

def load_trades():
    """Load trades from JSON file"""
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, 'r', encoding='utf-8') as f:
            trades_data = json.load(f)
        
        for trade in trades_data:
            if 'images' in trade and trade['images']:
                decoded_images = []
                for img_b64 in trade['images']:
                    try:
                        img_data = base64.b64decode(img_b64)
                        img = Image.open(io.BytesIO(img_data))
                        decoded_images.append(img)
                    except:
                        decoded_images.append(None)
                trade['images'] = decoded_images
            else:
                trade['images'] = []
        
        st.session_state.trades = trades_data
        return len(trades_data)
    return 0

def add_trade(trade_data):
    st.session_state.trades.append(trade_data)
    save_trades()

def update_trade(index, trade_data):
    if 0 <= index < len(st.session_state.trades):
        st.session_state.trades[index] = trade_data
        save_trades()

def delete_trade(index):
    if 0 <= index < len(st.session_state.trades):
        del st.session_state.trades[index]
        save_trades()

def calculate_pnl_percent(trade):
    """Calculate actual P&L percentage with leverage"""
    if trade['win_loss'] == 'Win':
        return trade.get('reward_percent', 0) * trade.get('leverage', 1)
    elif trade['win_loss'] == 'Loss':
        return -trade.get('risk_percent', 0) * trade.get('leverage', 1)
    else:
        return 0

def calculate_performance_metrics(trades, start_capital=1000):
    """Calculate ALL performance metrics including drawdown"""
    if not trades:
        return {}
    
    closed_trades = [t for t in trades if t.get('win_loss') in ['Win', 'Loss']]
    wins = [t for t in closed_trades if t['win_loss'] == 'Win']
    losses = [t for t in closed_trades if t['win_loss'] == 'Loss']
    
    total_trades = len(closed_trades)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    # Win/Loss amounts
    win_amounts = [calculate_pnl_percent(t) for t in wins]
    loss_amounts = [calculate_pnl_percent(t) for t in losses]
    
    avg_win = sum(win_amounts) / win_count if win_count > 0 else 0
    avg_loss = sum(loss_amounts) / loss_count if loss_count > 0 else 0
    largest_win = max(win_amounts) if win_amounts else 0
    largest_loss = min(loss_amounts) if loss_amounts else 0
    
    # R:R metrics
    rr_ratios = [t.get('rr_ratio', 0) for t in closed_trades]
    avg_rr = sum(rr_ratios) / total_trades if total_trades > 0 else 0
    best_rr = max(rr_ratios) if rr_ratios else 0
    worst_rr = min(rr_ratios) if rr_ratios else 0
    
    # Profit Factor
    gross_profit = sum(win_amounts)
    gross_loss = abs(sum(loss_amounts))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Expectancy
    expectancy = (avg_win * win_rate / 100) - (abs(avg_loss) * (1 - win_rate / 100))
    expectancy_dollars = expectancy * start_capital / 100
    
    # Consecutive wins/losses
    streaks = []
    current_streak = 0
    for trade in closed_trades:
        if trade['win_loss'] == 'Win':
            if current_streak >= 0:
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 1
        else:
            if current_streak <= 0:
                current_streak -= 1
            else:
                streaks.append(current_streak)
                current_streak = -1
    if current_streak != 0:
        streaks.append(current_streak)
    
    max_win_streak = max([s for s in streaks if s > 0], default=0)
    max_loss_streak = abs(min([s for s in streaks if s < 0], default=0))
    
    # Equity curve and drawdown calculation
    equity = start_capital
    equity_curve = [equity]
    peaks = [equity]
    
    for trade in closed_trades:
        pnl_pct = calculate_pnl_percent(trade)
        equity = equity * (1 + pnl_pct / 100.0)
        equity_curve.append(equity)
        peaks.append(max(peaks[-1], equity))
    
    # Calculate drawdowns
    drawdowns = [(peaks[i] - equity_curve[i]) / peaks[i] * 100 for i in range(len(equity_curve))]
    max_drawdown = max(drawdowns) if drawdowns else 0
    max_drawdown_amount = max([(peaks[i] - equity_curve[i]) for i in range(len(equity_curve))], default=0)
    
    # Average drawdown on losing trades
    avg_loss_drawdown = abs(avg_loss) if avg_loss != 0 else 0
    
    # Recovery factor (total profit / max drawdown)
    total_return = (equity_curve[-1] - start_capital)
    recovery_factor = total_return / max_drawdown_amount if max_drawdown_amount > 0 else 0
    
    # Sharpe ratio approximation
    returns = [calculate_pnl_percent(t) for t in closed_trades]
    sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 1 and np.std(returns) > 0 else 0
    
    # Risk of Ruin (simplified)
    if win_rate > 0 and avg_win > 0 and avg_loss < 0:
        risk_of_ruin = ((1 - win_rate/100) / (win_rate/100)) ** (start_capital / (abs(avg_loss) * start_capital / 100))
        risk_of_ruin = min(risk_of_ruin, 100)
    else:
        risk_of_ruin = 100
    
    return {
        # Basic metrics
        'total_trades': total_trades,
        'wins': win_count,
        'losses': loss_count,
        'win_rate': round(win_rate, 2),
        'avg_win_pct': round(avg_win, 2),
        'avg_loss_pct': round(avg_loss, 2),
        'largest_win_pct': round(largest_win, 2),
        'largest_loss_pct': round(largest_loss, 2),
        
        # R:R metrics
        'avg_rr': round(avg_rr, 2),
        'best_rr': round(best_rr, 2),
        'worst_rr': round(worst_rr, 2),
        
        # Streaks
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        
        # Risk metrics
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 0,
        'expectancy_r': round(expectancy, 2),
        'expectancy_dollars': round(expectancy_dollars, 2),
        
        # Drawdown
        'max_drawdown_pct': round(max_drawdown, 2),
        'max_drawdown_amount': round(max_drawdown_amount, 2),
        'avg_loss_drawdown': round(avg_loss_drawdown, 2),
        
        # Advanced metrics
        'recovery_factor': round(recovery_factor, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'risk_of_ruin': round(risk_of_ruin, 2),
        
        # Equity data for charts
        'equity_curve': equity_curve,
        'drawdowns': drawdowns,
        'total_return_pct': round((equity_curve[-1] - start_capital) / start_capital * 100, 2),
        'final_equity': round(equity_curve[-1], 2),
    }

# Load existing trades
loaded_count = load_trades()
if loaded_count > 0:
    st.sidebar.success(f"✅ Loaded {loaded_count} trades from {TRADES_FILE}")

# ----- MAIN LAYOUT -----
st.title("📈 Trading Dashboard")
st.markdown("---")

# ----- SIDEBAR - ADD/EDIT TRADE -----
with st.sidebar:
    st.header("✏️ Add New Trade")
    
    editing = st.session_state.edit_index is not None
    if editing:
        st.success(f"Editing Trade #{st.session_state.edit_index + 1}")
        trade_to_edit = st.session_state.trades[st.session_state.edit_index]
    else:
        trade_to_edit = None
    
    with st.form("trade_form", clear_on_submit=not editing):
        # Date & Time
        col1, col2 = st.columns(2)
        with col1:
            trade_date = st.date_input("📅 Date", value=datetime.now().date())
        with col2:
            trade_time = st.time_input("⏰ Time", value=datetime.now().time())
        
        # Symbol & Timeframe
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("🔖 Symbol", value=trade_to_edit.get('symbol', 'BTCUSDT') if editing else 'BTCUSDT')
        with col2:
            timeframe = st.selectbox("⏱️ Timeframe", ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"], index=2)
        
        # Direction & Leverage
        col1, col2 = st.columns(2)
        with col1:
            direction = st.selectbox("📊 Direction", ["Long", "Short"], 
                                     index=0 if not editing else (0 if trade_to_edit.get('direction', 'Long') == "Long" else 1))
        with col2:
            leverage = st.number_input("⚡ Leverage", min_value=1, max_value=100, 
                                      value=trade_to_edit.get('leverage', 1) if editing else 1, step=1)
        
        # Prices
        col1, col2 = st.columns(2)
        with col1:
            entry_price = st.number_input("💲 Entry Price", min_value=0.0, 
                                         value=trade_to_edit.get('price_entry', 0.0) if editing else 0.0, 
                                         step=0.5, format="%.2f")
        with col2:
            sl_price = st.number_input("🛑 Stop Loss", min_value=0.0, 
                                      value=trade_to_edit.get('sl_price', 0.0) if editing else 0.0, 
                                      step=0.5, format="%.2f")
        
        # Result
        win_loss = st.selectbox("🏆 Result", ["Win", "Loss", "Breakeven", "Open"],
                               index=0 if not editing else (["Win", "Loss", "Breakeven", "Open"].index(trade_to_edit.get('win_loss', 'Win'))))
        
        # TP Setup
        st.markdown("---")
        st.subheader("🎯 Take Profit Setup")
        
        tp_type = st.selectbox("TP Type", ["Full TPs", "Half", "Risk Free"],
                               index=0 if not editing else (["Full TPs", "Half", "Risk Free"].index(trade_to_edit.get('tp_type', 'Full TPs'))))
        
        tp1 = 0.0
        tp2 = 0.0
        
        if tp_type == "Full TPs":
            col1, col2 = st.columns(2)
            with col1:
                tp1 = st.number_input("🎯 TP1", min_value=0.0, step=0.5, format="%.2f",
                                      value=trade_to_edit.get('tp1', 0.0) if editing else 0.0)
            with col2:
                tp2 = st.number_input("🎯 TP2", min_value=0.0, step=0.5, format="%.2f",
                                      value=trade_to_edit.get('tp2', 0.0) if editing else 0.0)
        elif tp_type == "Half":
            tp1 = st.number_input("🎯 TP1 (Only)", min_value=0.0, step=0.5, format="%.2f",
                                 value=trade_to_edit.get('tp1', 0.0) if editing else 0.0)
        
        # Risk / Reward (Manual Entry)
        st.markdown("---")
        st.subheader("📐 Risk / Reward (Manual)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_percent = st.number_input("⚠️ Risk %", min_value=0.0, max_value=100.0, step=0.1, format="%.2f",
                                          value=trade_to_edit.get('risk_percent', 0.0) if editing else 0.0)
        with col2:
            reward_percent = st.number_input("🎯 Reward %", min_value=0.0, max_value=100.0, step=0.1, format="%.2f",
                                            value=trade_to_edit.get('reward_percent', 0.0) if editing else 0.0)
        with col3:
            rr_ratio = st.number_input("📊 R:R", min_value=0.0, step=0.01, format="%.2f",
                                      value=trade_to_edit.get('rr_ratio', 0.0) if editing else 0.0)
        
        # Notes & Images
        st.markdown("---")
        notes = st.text_area("📝 Why did you take this trade?", height=80,
                            value=trade_to_edit.get('notes', '') if editing else "")
        
        st.subheader("🖼️ Screenshots (Max 4)")
        uploaded_images = []
        for i in range(4):
            img = st.file_uploader(f"Image {i+1}", type=['png', 'jpg', 'jpeg'], key=f"img_{i}_{editing}")
            if img is not None:
                uploaded_images.append(Image.open(img))
        
        if editing and trade_to_edit.get('images') and not uploaded_images:
            uploaded_images = trade_to_edit['images']
        
        # Psychology
        st.markdown("---")
        st.subheader("🧠 Psychology & Self-Reflection")
        
        emotion_options = ["😨 Fearful", "😐 Neutral", "😌 Calm", "😃 Confident", "🤯 Overconfident"]
        pre_emotion = st.select_slider("😌 How did you feel BEFORE entering?", options=emotion_options,
                                       value=trade_to_edit.get('pre_emotion', "😐 Neutral") if editing else "😐 Neutral")
        
        rules_followed = st.slider("📋 Did you follow your trading rules? (1 = No, 10 = Perfectly)",
                                   min_value=1, max_value=10, 
                                   value=trade_to_edit.get('rules_followed', 7) if editing else 7)
        
        mistake_options = ["None", "Entered too early", "Exited too early", "Exited too late", 
                          "Didn't respect SL", "Revenge trade", "Oversized position", "FOMO", 
                          "Chasing price", "No confirmation"]
        mistake = st.selectbox("🪞 What was your biggest mistake (if any)?", options=mistake_options,
                              index=mistake_options.index(trade_to_edit.get('mistake', 'None')) if editing and trade_to_edit.get('mistake', 'None') in mistake_options else 0)
        
        lesson = st.text_input("💡 One thing you learned from this trade",
                              placeholder="e.g., 'Wait for confirmation' or 'Trust my SL'",
                              value=trade_to_edit.get('lesson', '') if editing else "")
        
        # Submit
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("💾 Save Trade", use_container_width=True, type="primary")
        with col2:
            if editing:
                cancel_edit = st.form_submit_button("❌ Cancel Edit", use_container_width=True)
                if cancel_edit:
                    st.session_state.edit_index = None
                    st.rerun()
    
    if submitted:
        trade_data = {
            'date': str(trade_date),
            'time': str(trade_time),
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': direction,
            'leverage': leverage,
            'price_entry': entry_price,
            'sl_price': sl_price,
            'win_loss': win_loss,
            'tp_type': tp_type,
            'tp1': tp1,
            'tp2': tp2 if tp_type == "Full TPs" else 0.0,
            'risk_percent': risk_percent,
            'reward_percent': reward_percent,
            'rr_ratio': rr_ratio,
            'notes': notes,
            'images': uploaded_images if uploaded_images else [],
            'created_at': datetime.now().isoformat(),
            'pre_emotion': pre_emotion,
            'rules_followed': rules_followed,
            'mistake': mistake,
            'lesson': lesson
        }
        
        if editing:
            update_trade(st.session_state.edit_index, trade_data)
            st.session_state.edit_index = None
            st.success("✅ Trade Updated Successfully!")
        else:
            add_trade(trade_data)
            st.success("✅ Trade Added Successfully!")
        
        st.rerun()

# ----- TRADE LOG -----
st.subheader(f"📋 Trade Log ({len(st.session_state.trades)} trades)")

if not st.session_state.trades:
    st.info("👋 No trades yet. Add your first trade from the sidebar!")
else:
    for idx, trade in enumerate(st.session_state.trades):
        with st.container():
            col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1, 2, 1, 1, 1, 1, 1, 1])
            with col1:
                st.write(f"**#{idx+1}**")
            with col2:
                st.write(f"📅 {trade['date']} ⏰ {trade['time']}")
                st.caption(f"{trade.get('symbol', 'BTCUSDT')} | {trade.get('timeframe', '5m')}")
            with col3:
                if trade['direction'] == "Long":
                    st.markdown(f"<h3 style='color:green;margin:0'>▲ LONG</h3>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 style='color:red;margin:0'>▼ SHORT</h3>", unsafe_allow_html=True)
            with col4:
                st.metric("Entry", f"${trade['price_entry']:.2f}")
            with col5:
                st.metric("Leverage", f"{trade.get('leverage', 1)}x")
            with col6:
                wl = trade['win_loss']
                if wl == "Win":
                    st.markdown(f"<h3 style='color:green;margin:0'>🏆 WIN</h3>", unsafe_allow_html=True)
                elif wl == "Loss":
                    st.markdown(f"<h3 style='color:red;margin:0'>❌ LOSS</h3>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 style='color:blue;margin:0'>🔄 {wl}</h3>", unsafe_allow_html=True)
            with col7:
                st.write(f"🎯 {trade['tp_type']}")
                if trade['tp1'] > 0:
                    st.caption(f"TP1: {trade['tp1']:.2f}")
                if trade.get('tp2', 0) > 0:
                    st.caption(f"TP2: {trade['tp2']:.2f}")
            with col8:
                edit_btn = st.button("✏️ Edit", key=f"edit_{idx}")
                delete_btn = st.button("🗑️ Delete", key=f"del_{idx}")
                if edit_btn:
                    st.session_state.edit_index = idx
                    st.rerun()
                if delete_btn:
                    delete_trade(idx)
                    st.rerun()
        
        with st.expander(f"📝 Details - Trade #{idx+1}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**🛑 Stop Loss:** ${trade['sl_price']:.2f}")
                st.markdown(f"**⚠️ Risk %:** {trade.get('risk_percent', 0):.2f}%")
                st.markdown(f"**🎯 Reward %:** {trade.get('reward_percent', 0):.2f}%")
                st.markdown(f"**📊 R:R:** 1:{trade.get('rr_ratio', 0):.2f}")
            with col2:
                if trade.get('notes'):
                    st.markdown("**📝 Entry Rationale:**")
                    st.info(trade['notes'])
            
            if trade.get('pre_emotion'):
                st.markdown("---")
                st.markdown("**🧠 Psychology**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Emotion:** {trade['pre_emotion']}")
                with col2:
                    st.write(f"**Rules:** {trade.get('rules_followed', 'N/A')}/10")
                with col3:
                    st.write(f"**Mistake:** {trade.get('mistake', 'None')}")
                if trade.get('lesson'):
                    st.info(f"💡 **Lesson:** {trade['lesson']}")
            
            if trade.get('images') and any(trade['images']):
                valid_images = [img for img in trade['images'] if img is not None]
                if valid_images:
                    st.markdown("**🖼️ Screenshots:**")
                    img_cols = st.columns(len(valid_images))
                    for img_idx, img in enumerate(valid_images):
                        with img_cols[img_idx]:
                            st.image(img, use_container_width=True)
        
        st.markdown("---")

# ========== ADVANCED PERFORMANCE METRICS ==========
if st.session_state.trades:
    metrics = calculate_performance_metrics(st.session_state.trades, st.session_state.start_capital)
    
    st.subheader("📊 Advanced Performance Analytics")
    
    # Row 1: Basic Stats
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📊 Total Trades", metrics.get('total_trades', 0))
    with col2:
        st.metric("🏆 Wins", metrics.get('wins', 0))
    with col3:
        st.metric("❌ Losses", metrics.get('losses', 0))
    with col4:
        win_rate = metrics.get('win_rate', 0)
        st.metric("🎯 Win Rate", f"{win_rate:.1f}%", 
                  delta=f"{win_rate - 50:.1f}%" if win_rate != 50 else None)
    with col5:
        st.metric("📈 Avg R:R", f"1:{metrics.get('avg_rr', 0):.2f}")
    
    st.markdown("---")
    
    # Row 2: Win/Loss Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Avg Win", f"+{metrics.get('avg_win_pct', 0):.2f}%")
    with col2:
        st.metric("📉 Avg Loss", f"{metrics.get('avg_loss_pct', 0):.2f}%")
    with col3:
        st.metric("🏆 Largest Win", f"+{metrics.get('largest_win_pct', 0):.2f}%")
    with col4:
        st.metric("💀 Largest Loss", f"{metrics.get('largest_loss_pct', 0):.2f}%")
    
    st.markdown("---")
    
    # Row 3: Risk Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Profit Factor", f"{metrics.get('profit_factor', 0):.2f}x")
    with col2:
        st.metric("💰 Expectancy", f"{metrics.get('expectancy_dollars', 0):.2f}$")
    with col3:
        st.metric("⚡ Best R:R", f"1:{metrics.get('best_rr', 0):.2f}")
    with col4:
        st.metric("🐌 Worst R:R", f"1:{metrics.get('worst_rr', 0):.2f}")
    
    st.markdown("---")
    
    # Row 4: Streaks & Drawdown
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔥 Max Win Streak", metrics.get('max_win_streak', 0))
    with col2:
        st.metric("💔 Max Loss Streak", metrics.get('max_loss_streak', 0))
    with col3:
        st.metric("📉 Max Drawdown", f"-{metrics.get('max_drawdown_pct', 0):.2f}%", 
                  delta=f"-${metrics.get('max_drawdown_amount', 0):.2f}")
    with col4:
        st.metric("💰 Avg Loss Drawdown", f"-{metrics.get('avg_loss_drawdown', 0):.2f}%")
    
    st.markdown("---")
    
    # # Row 5: Advanced Metrics
    # col1, col2, col3, col4 = st.columns(4)
    # with col1:
    #     st.metric("🔄 Recovery Factor", f"{metrics.get('recovery_factor', 0):.2f}x")
    # with col2:
    #     st.metric("📐 Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    # with col3:
    #     risk_of_ruin = metrics.get('risk_of_ruin', 100)
    #     st.metric("⚠️ Risk of Ruin", f"{risk_of_ruin:.1f}%")
    # with col4:
    #     st.metric("📈 Total Return", f"{metrics.get('total_return_pct', 0):+.2f}%")
    
    # st.markdown("---")
    
    # Psychology Insights
    closed_trades = [t for t in st.session_state.trades if t.get('win_loss') in ['Win', 'Loss']]
    if len(closed_trades) >= 3:
        st.subheader("🧠 Psychology Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            mistakes = [t.get('mistake') for t in closed_trades if t.get('mistake') and t.get('mistake') != 'None']
            if mistakes:
                common_mistake = Counter(mistakes).most_common(1)[0]
                st.warning(f"⚠️ Most common mistake: **{common_mistake[0]}** ({common_mistake[1]} times)")
            
            # Win rate by emotion
            wins_by_emotion = {}
            for trade in closed_trades:
                emotion = trade.get('pre_emotion', 'Unknown')
                if emotion not in wins_by_emotion:
                    wins_by_emotion[emotion] = {'wins': 0, 'total': 0}
                wins_by_emotion[emotion]['total'] += 1
                if trade['win_loss'] == 'Win':
                    wins_by_emotion[emotion]['wins'] += 1
            
            if wins_by_emotion:
                st.write("**Win rate by emotion:**")
                for emotion, data in wins_by_emotion.items():
                    rate = data['wins'] / data['total'] * 100
                    st.caption(f"{emotion}: {rate:.0f}% ({data['wins']}/{data['total']})")
        
        with col2:
            discipline_scores = [t.get('rules_followed') for t in closed_trades if t.get('rules_followed')]
            if discipline_scores:
                avg_discipline = sum(discipline_scores) / len(discipline_scores)
                if avg_discipline >= 8:
                    st.success(f"🎯 Excellent discipline! **{avg_discipline:.1f}/10**")
                elif avg_discipline >= 6:
                    st.info(f"📋 Good discipline: **{avg_discipline:.1f}/10**")
                else:
                    st.warning(f"⚠️ Discipline needs work: **{avg_discipline:.1f}/10**")
                
                # Correlation between discipline and win rate
                high_discipline_wins = sum(1 for t in closed_trades if t.get('rules_followed', 0) >= 8 and t['win_loss'] == 'Win')
                high_discipline_total = sum(1 for t in closed_trades if t.get('rules_followed', 0) >= 8)
                if high_discipline_total > 0:
                    high_rate = high_discipline_wins / high_discipline_total * 100
                    st.caption(f"📊 Win rate when discipline ≥ 8: {high_rate:.0f}%")

# ========== EQUITY & DRAWDOWN CHARTS ==========
if st.session_state.trades:
    st.markdown("---")
    st.subheader("💰 Portfolio Performance")
    
    col_cap, col_curr, col_return = st.columns(3)
    with col_cap:
        new_capital = st.number_input("💵 Starting Capital ($)", min_value=100.0,
                                      value=st.session_state.start_capital, step=100.0)
        if new_capital != st.session_state.start_capital:
            st.session_state.start_capital = new_capital
            st.rerun()
    
    # Recalculate metrics with new capital
    metrics = calculate_performance_metrics(st.session_state.trades, st.session_state.start_capital)
    
    with col_curr:
        st.metric("📈 Current Equity", f"${metrics.get('final_equity', st.session_state.start_capital):,.2f}")
    with col_return:
        st.metric("📊 Total Return %", f"{metrics.get('total_return_pct', 0):+.2f}%", delta_color="normal")
    
    # Equity Curve Chart
    if metrics.get('equity_curve'):
        st.markdown("**📉 Equity Curve**")
        chart_data = pd.DataFrame({
            "Trade #": range(len(metrics['equity_curve'])),
            "Equity ($)": metrics['equity_curve']
        }).set_index("Trade #")
        st.line_chart(chart_data)
        
        # Drawdown Chart
        if metrics.get('drawdowns'):
            st.markdown("**📉 Drawdown %**")
            drawdown_data = pd.DataFrame({
                "Trade #": range(len(metrics['drawdowns'])),
                "Drawdown %": metrics['drawdowns']
            }).set_index("Trade #")
            st.area_chart(drawdown_data)
        
        st.caption("💡 *Calculated by applying each trade's Risk% (loss) or Reward% (win) multiplied by leverage.*")

st.markdown("---")
st.caption(f"📁 Data saved to: {TRADES_FILE} | Total trades: {len(st.session_state.trades)}")