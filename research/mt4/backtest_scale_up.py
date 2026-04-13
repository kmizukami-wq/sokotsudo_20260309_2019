#!/usr/bin/env python3
"""
BB_Reversal_Martin ロット拡大戦略検証
FXTF手数料込みで4モード比較: Rank1 vs 2x vs 5x vs 証拠金MAX
"""
import numpy as np
import pandas as pd
import yfinance as yf

# EA パラメータ
RISK_PCT=0.8; RR_RATIO=2.0; BE_TRIGGER_RR=1.0; PARTIAL_RR=1.5; PARTIAL_PCT=0.5
TRAIL_ATR_MULT=0.5; MAX_HOLD_BARS=20; MARTIN=[1.0,1.5,2.0]
SL_MULTS={1:2.0,2:1.8,3:1.5}; ATR_FILTER_MULT=2.5
INITIAL_EQUITY=500000

# FXTF手数料体系
FXTF_FEE = {
    "USDJPY":(10000, 40), "EURJPY":(10000, 60), "EURUSD":(10000, 60),
    "AUDJPY":(10000, 60), "GBPJPY":(10000, 80),
    "GBPUSD":( 5000,100), "AUDUSD":( 5000, 70),
    "NZDJPY":( 3000, 70), "CADJPY":( 3000, 60), "USDCAD":( 3000,120),
    "ZARJPY":( 1000, 70), "EURAUD":( 1000,220), "NZDUSD":( 1000,150),
    "EURGBP":( 1000,180), "GBPAUD":( 1000,240), "AUDNZD":( 1000,120),
    "CHFJPY":( 1000,180), "USDCHF":( 1000,100), "EURNZD":( 1000,290),
    "GBPCAD":( 1000,210), "AUDCHF":( 1000,200), "EURCAD":( 1000,170),
    "EURCHF":( 1000,220), "GBPNZD":( 1000,290), "GBPCHF":( 1000,300),
    "AUDCAD":( 1000,180),
}

# 稼働対象12ペアのみ
TARGET_PAIRS = {
    "USDJPY":"USDJPY=X","EURJPY":"EURJPY=X","EURUSD":"EURUSD=X",
    "GBPJPY":"GBPJPY=X","AUDJPY":"AUDJPY=X",
    "AUDUSD":"AUDUSD=X","USDCAD":"USDCAD=X",
    "EURAUD":"EURAUD=X","GBPAUD":"GBPAUD=X","NZDCHF":"NZDCHF=X",
    "EURCHF":"EURCHF=X","EURNZD":"EURNZD=X","GBPCHF":"GBPCHF=X","AUDCHF":"AUDCHF=X",
}

# 4モードのMaxLot定義
MODES = {
    "A:Rank1cap(現行)": {
        "USDJPY":0.10,"EURJPY":0.10,"EURUSD":0.10,"GBPJPY":0.10,"AUDJPY":0.10,
        "AUDUSD":0.05,"USDCAD":0.03,
        "EURAUD":0.01,"GBPAUD":0.01,"NZDCHF":0.01,"EURCHF":0.01,"EURNZD":0.01,
        "GBPCHF":0.01,"AUDCHF":0.01,
    },
    "B:Premium2倍": {
        "USDJPY":0.20,"EURJPY":0.20,"EURUSD":0.20,"GBPJPY":0.15,"AUDJPY":0.20,
        "AUDUSD":0.10,"USDCAD":0.05,
        "EURAUD":0.01,"GBPAUD":0.01,"NZDCHF":0.01,"EURCHF":0.01,"EURNZD":0.01,
        "GBPCHF":0.01,"AUDCHF":0.01,
    },
    "C:Premium5倍": {
        "USDJPY":0.50,"EURJPY":0.50,"EURUSD":0.50,"GBPJPY":0.30,"AUDJPY":0.50,
        "AUDUSD":0.25,"USDCAD":0.10,
        "EURAUD":0.01,"GBPAUD":0.01,"NZDCHF":0.01,"EURCHF":0.01,"EURNZD":0.01,
        "GBPCHF":0.01,"AUDCHF":0.01,
    },
    "D:証拠金MAX(1M想定)": {  # 資金1M想定での最大
        "USDJPY":0.30,"EURJPY":0.30,"EURUSD":0.30,"GBPJPY":0.20,"AUDJPY":0.30,
        "AUDUSD":0.15,"USDCAD":0.08,
        "EURAUD":0.01,"GBPAUD":0.01,"NZDCHF":0.01,"EURCHF":0.01,"EURNZD":0.01,
        "GBPCHF":0.01,"AUDCHF":0.01,
    },
}

# ペア別証拠金（Mar金使用率計算用、4%マージン）
# Notional × 0.04 / ロット
APPROX_RATES = {
    "USDJPY":158, "EURJPY":170, "EURUSD":1.085, "GBPJPY":205, "AUDJPY":104,
    "AUDUSD":0.65, "USDCAD":1.40, "EURAUD":1.64, "GBPAUD":1.94,
    "NZDCHF":0.50, "EURCHF":0.90, "EURNZD":1.85, "GBPCHF":1.10, "AUDCHF":0.55,
}
USDJPY_RATE = 158  # クロス円換算用

def calc_margin_yen(pair, lots):
    """FXTF 4%マージン、円換算の必要証拠金"""
    units = lots * 100000
    rate = APPROX_RATES.get(pair, 1.0)
    if pair.endswith("JPY"):
        notional_yen = units * rate
    else:
        # 基軸通貨 × USDJPY で円換算
        # For USDxxx pairs, base is USD → units × USDJPY
        # For EURxxx/GBPxxx/AUDxxx/NZDxxx USD, base × USD × USDJPY
        base = pair[:3]
        if base == "USD":
            notional_yen = units * USDJPY_RATE
        elif base == "EUR":
            notional_yen = units * 1.08 * USDJPY_RATE
        elif base == "GBP":
            notional_yen = units * 1.27 * USDJPY_RATE
        elif base == "AUD":
            notional_yen = units * 0.65 * USDJPY_RATE
        elif base == "NZD":
            notional_yen = units * 0.60 * USDJPY_RATE
        else:
            notional_yen = units * USDJPY_RATE
    return notional_yen * 0.04

def pcfg(name):
    j=name.endswith("JPY")
    if j:tv=1000
    elif name in("EURUSD","GBPUSD","AUDUSD","NZDUSD"):tv=1500
    else:tv=1200
    return{"pip":0.01 if j else 0.0001,"pip_mult":100 if j else 10000,"tick_value":tv}

def fetch(ticker):
    df=yf.download(ticker,period="60d",interval="15m",progress=False)
    if df.empty:return None
    df=df.droplevel("Ticker",axis=1) if isinstance(df.columns,pd.MultiIndex) else df
    df=df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close"})
    df=df[["open","high","low","close"]].dropna()
    df["time"]=df.index;df=df.reset_index(drop=True)
    return df

def calc_ind(df):
    c=df["close"]
    df["sma200"]=c.rolling(200).mean();df["sma50"]=c.rolling(50).mean();df["sma20"]=c.rolling(20).mean()
    df["sma200_up"]=df["sma200"]>df["sma200"].shift(5);df["sma50_up"]=df["sma50"]>df["sma50"].shift(5)
    d=c.diff();g=d.clip(lower=0).ewm(span=10,adjust=False).mean()
    l=(-d.clip(upper=0)).ewm(span=10,adjust=False).mean()
    df["rsi"]=100-(100/(1+g/l.replace(0,np.nan)))
    tr=pd.concat([df["high"]-df["low"],(df["high"]-c.shift(1)).abs(),(df["low"]-c.shift(1)).abs()],axis=1).max(axis=1)
    df["atr14"]=tr.rolling(14).mean();df["atr14_ma100"]=df["atr14"].rolling(100).mean()
    bm=c.rolling(20).mean();bs=c.rolling(20).std()
    df["bb_upper"]=bm+2.5*bs;df["bb_lower"]=bm-2.5*bs
    fm=c.rolling(10).mean();fs=c.rolling(10).std()
    df["fbb_upper"]=fm+2.0*fs;df["fbb_lower"]=fm-2.0*fs
    return df.dropna().reset_index(drop=True)

def check_sig(row,prev):
    if row["atr14"]>=row["atr14_ma100"]*ATR_FILTER_MULT:return 0
    c1,c2,o1,rsi=row["close"],prev["close"],row["open"],row["rsi"]
    u2,u5=row["sma200_up"],row["sma50_up"]
    if u2 and c2<=prev["bb_lower"] and c1>row["bb_lower"] and c1>o1 and rsi<42:return 1
    if not u2 and c2>=prev["bb_upper"] and c1<row["bb_upper"] and c1<o1 and rsi>58:return -1
    if u2 and u5 and c1<=row["fbb_lower"] and rsi<48:return 2
    if not u2 and not u5 and c1>=row["fbb_upper"] and rsi>52:return -2
    gap=abs(row["sma20"]-row["sma50"])
    if gap>=row["atr14"]*2:
        lo,hi=min(row["sma20"],row["sma50"]),max(row["sma20"],row["sma50"])
        if u2 and lo<=c1<=hi and 30<=rsi<=50:return 3
        if not u2 and lo<=c1<=hi and 50<=rsi<=70:return -3
    return 0

def calc_fee(pair,lots):
    if pair not in FXTF_FEE:return 0
    rank1,r2=FXTF_FEE[pair]
    units=lots*100000
    if units<=rank1:return 0
    return units/10000*r2

def run_bt(df,cfg,pair,max_lot):
    pm=cfg["pip_mult"];tv=cfg["tick_value"];pu=cfg["pip"]
    eq=float(INITIAL_EQUITY);ep=eq;mdd=0.0;mseq=eq
    ms=0;cl=0;trades=[];monthly=[];cm=None
    i=1
    while i<len(df)-MAX_HOLD_BARS-1:
        row=df.iloc[i];prev=df.iloc[i-1]
        rm=pd.to_datetime(row["time"]).month
        if cm is None:cm=rm;mseq=eq
        elif rm!=cm:monthly.append({"m":cm,"p":(eq-mseq)/mseq*100});cm=rm;mseq=eq
        sig=check_sig(row,prev)
        if sig==0:i+=1;continue
        d=1 if sig>0 else -1;st=abs(sig);atr=row["atr14"]
        sld=atr*SL_MULTS.get(st,1.5);tpd=sld*RR_RATIO
        ep_=row["close"];mm=MARTIN[min(ms,2)]

        # リスク%ベース計算 → MaxLotでcap
        ra=eq*RISK_PCT/100.0
        sp_pips=sld*pm
        if sp_pips<=0:i+=1;continue
        raw=(ra*mm)/(sp_pips*tv)
        lots=max(0.01,int(raw/0.01)*0.01)
        lots=min(lots,max_lot)

        slp=ep_-d*sld;tpp=ep_+d*tpd
        be=False;pc=False;rl=lots;rp=0.0;res="timeout";eb=i;pnl=0.0;to=False
        fee_entry=calc_fee(pair,lots)

        for j in range(1,MAX_HOLD_BARS+1):
            if i+j>=len(df):break
            b=df.iloc[i+j];ba=b["atr14"] if not np.isnan(b["atr14"]) else atr
            if d==1:
                if not be and(b["high"]-ep_)>=sld*BE_TRIGGER_RR:slp=ep_;be=True
                if not pc and(b["high"]-ep_)>=sld*PARTIAL_RR:
                    cl_=round(rl*PARTIAL_PCT,2)
                    if cl_>=0.01 and(rl-cl_)>=0.01:
                        rp+=(sld*PARTIAL_RR)*pm*cl_*tv;rl=round(rl-cl_,2);pc=True
                        ts=ep_+sld*PARTIAL_RR-ba*TRAIL_ATR_MULT
                        if ts>slp:slp=ts
                if pc:
                    ts=b["high"]-ba*TRAIL_ATR_MULT
                    if ts>slp:slp=ts
                if b["low"]<=slp:pnl=(slp-ep_)*pm*rl*tv+rp;res="BE" if be else "SL";eb=i+j;break
                if b["high"]>=tpp:pnl=tpd*pm*rl*tv+rp;res="TP";eb=i+j;break
            else:
                if not be and(ep_-b["low"])>=sld*BE_TRIGGER_RR:slp=ep_;be=True
                if not pc and(ep_-b["low"])>=sld*PARTIAL_RR:
                    cl_=round(rl*PARTIAL_PCT,2)
                    if cl_>=0.01 and(rl-cl_)>=0.01:
                        rp+=(sld*PARTIAL_RR)*pm*cl_*tv;rl=round(rl-cl_,2);pc=True
                        ts=ep_-sld*PARTIAL_RR+ba*TRAIL_ATR_MULT
                        if ts<slp:slp=ts
                if pc:
                    ts=b["low"]+ba*TRAIL_ATR_MULT
                    if ts<slp:slp=ts
                if b["high"]>=slp:pnl=(ep_-slp)*pm*rl*tv+rp;res="BE" if be else "SL";eb=i+j;break
                if b["low"]<=tpp:pnl=tpd*pm*rl*tv+rp;res="TP";eb=i+j;break
        else:
            ex=df.iloc[min(i+MAX_HOLD_BARS,len(df)-1)]["close"]
            pnl=d*(ex-ep_)*pm*rl*tv+rp;to=True

        # 手数料(往復)
        fee_round = fee_entry * 2
        if pc: fee_round += calc_fee(pair,lots*PARTIAL_PCT)
        pnl -= fee_round

        if to or res=="BE":cl=0;ms=0
        elif pnl<0:
            cl+=1
            if cl>=3:ms=0;cl=0
            else:ms=min(cl,2)
        else:cl=0;ms=0
        eq+=pnl;ep=max(ep,eq);dd=(ep-eq)/ep*100 if ep>0 else 0;mdd=max(mdd,dd)
        trades.append({"result":res,"pnl":pnl,"lots":lots,"fee":fee_round})
        i=eb+1
    if mseq>0 and cm is not None:monthly.append({"m":cm,"p":(eq-mseq)/mseq*100})
    return trades,mdd,monthly,eq

def summarize(trades,mdd,monthly):
    if not trades:return None
    dt=pd.DataFrame(trades);n=len(dt)
    w=dt[dt["pnl"]>0];l=dt[dt["pnl"]<=0]
    wr=len(w)/n*100;tp=dt["pnl"].sum()
    gw=w["pnl"].sum() if len(w)>0 else 0;gl=abs(l["pnl"].sum()) if len(l)>0 else 0
    pf=gw/gl if gl>0 else float("inf")
    mm=pd.DataFrame(monthly)["p"].median() if monthly else 0
    avg_lot=dt["lots"].mean()
    total_fee=dt["fee"].sum()
    return{"n":n,"wr":wr,"pnl":tp,"pf":pf,"dd":mdd,"mm":mm,"lot":avg_lot,"fee":total_fee}

def main():
    print("="*115)
    print("BB_Reversal_Martin ロット拡大戦略検証（4モード比較）")
    print(f"初期: ¥{INITIAL_EQUITY:,} | RiskPct: {RISK_PCT}%")
    print("="*115)

    # データ取得
    all_data={}
    for pn,ticker in TARGET_PAIRS.items():
        print(f"  {pn:>7s}... ",end="",flush=True)
        df=fetch(ticker)
        if df is None or len(df)<300:print("SKIP");continue
        df=calc_ind(df)
        all_data[pn]=df
        print(f"{len(df)} bars")

    # 4モード実行
    results={}
    for mode_name, lots_map in MODES.items():
        print(f"\n[モード: {mode_name}]")
        for pn,df in all_data.items():
            cfg=pcfg(pn)
            max_lot=lots_map.get(pn, 0.01)
            tr,dd,mo,eq=run_bt(df,cfg,pn,max_lot)
            s=summarize(tr,dd,mo)
            results.setdefault(pn,{})[mode_name]=(s, max_lot)

    # 3モード比較テーブル: 純損益
    mode_keys=list(MODES.keys())
    print(f"\n{'='*115}")
    print(f"  【4モード比較: 純損益（手数料込み、3ヶ月）】")
    print(f"{'='*115}")
    print(f"  {'ペア':>7s}" + "".join(f"  {k:>22s}" for k in mode_keys))
    print(f"  {'-'*(7+24*len(mode_keys))}")
    totals={k:0 for k in mode_keys}
    for pn in all_data:
        row=f"  {pn:>7s}"
        for k in mode_keys:
            s,ml=results[pn][k]
            if s:
                row+=f"  PF{s['pf']:>4.2f} lot{ml:.2f} {s['pnl']:>+8,.0f}"
                totals[k]+=s["pnl"]
            else:
                row+=f"  {'---':>22s}"
        print(row)
    print(f"  {'-'*(7+24*len(mode_keys))}")
    tr=f"  {'合計':>7s}"
    for k in mode_keys:tr+=f"  {'':>14s}{totals[k]:>+8,.0f}"
    print(tr)

    # サマリー + 証拠金
    print(f"\n{'='*115}")
    print(f"  【全体サマリー & 必要証拠金】")
    print(f"{'='*115}")
    print(f"  {'モード':>22s}  {'合計損益(¥)':>13s}  {'月利中央値':>10s}  {'+ペア':>6s}  {'総手数料':>11s}  {'必要証拠金(¥)':>13s}  {'証拠金使用率':>12s}")
    print(f"  {'-'*110}")
    for mn in mode_keys:
        vals=[results[pn][mn][0] for pn in all_data if results[pn][mn][0]]
        if not vals:continue
        tot=sum(v["pnl"] for v in vals)
        mm=sum(v["mm"] for v in vals)
        plus=sum(1 for v in vals if v["pnl"]>0)
        tot_fee=sum(v["fee"] for v in vals)
        # 必要証拠金（全ペア同時想定）
        total_margin = sum(calc_margin_yen(pn, MODES[mn].get(pn, 0.01)) for pn in all_data)
        usage_rate = total_margin / INITIAL_EQUITY * 100
        print(f"  {mn:>22s}  {tot:>+12,.0f}  {mm:>+9.1f}%  {plus:>4d}/14  {tot_fee:>+10,.0f}  {total_margin:>+12,.0f}  {usage_rate:>10.1f}%")

    # 月間想定利益 (プラスペアのみ)
    print(f"\n{'='*115}")
    print(f"  【プラス収支ペアのみ採用 → 月間想定利益】")
    print(f"{'='*115}")
    print(f"  {'モード':>22s}  {'採用ペア':>8s}  {'3ヶ月損益':>12s}  {'月利合算':>10s}  {'月次絶対額':>12s}  {'ロスカット余裕':>13s}")
    print(f"  {'-'*100}")
    for mn in mode_keys:
        plus=[(pn, results[pn][mn][0]) for pn in all_data
              if results[pn][mn][0] and results[pn][mn][0]["pnl"]>0]
        if not plus:continue
        n=len(plus);pnl=sum(v[1]["pnl"] for v in plus)
        mm=sum(v[1]["mm"] for v in plus)
        abs_m=INITIAL_EQUITY*mm/100
        # 採用ペアだけの証拠金
        adopted_margin = sum(calc_margin_yen(pn, MODES[mn].get(pn, 0.01)) for pn,_ in plus)
        if adopted_margin < INITIAL_EQUITY:
            lc_room = (INITIAL_EQUITY - adopted_margin) / INITIAL_EQUITY * 100
            lc_status = f"{lc_room:>6.1f}% 余裕"
        else:
            lc_status = "証拠金不足!"
        print(f"  {mn:>22s}  {n:>6d}/14  {pnl:>+11,.0f}  {mm:>+9.1f}%  {abs_m:>+11,.0f}  {lc_status}")

    # 判定
    print(f"\n{'='*115}")
    print(f"  【推奨判定】")
    print(f"{'='*115}")
    best_mode = max(mode_keys, key=lambda m: totals[m])
    print(f"  全体損益最大: 【{best_mode}】 合計 ¥{totals[best_mode]:+,.0f}")
    print()
    print(f"  ※ただし証拠金使用率が100%超は危険（ロスカット確実）")
    print(f"  ※資金¥{INITIAL_EQUITY:,} では無理なロット拡大はできない")
    print(f"  ※¥1M以上なら案B(Premium2倍)、¥2M以上なら案C相当が可能")

main()
