#!/usr/bin/env python3
"""
BB_Reversal_Martin 固定ロット運用バックテスト
FXTF 1万通貨(0.1lot)までスプレッド0の優遇活用を想定
"""
import numpy as np
import pandas as pd
import yfinance as yf

# EAパラメータ
RISK_PCT=0.8; RR_RATIO=2.0; BE_TRIGGER_RR=1.0; PARTIAL_RR=1.5; PARTIAL_PCT=0.5
TRAIL_ATR_MULT=0.5; MAX_HOLD_BARS=20; MARTIN=[1.0,1.5,2.0]
SL_MULTS={1:2.0,2:1.8,3:1.5}; ATR_FILTER_MULT=2.5
INITIAL_EQUITY=500000

# 固定ロット運用設定
FIXED_LOT=0.1              # 10,000通貨

FXTF_PAIRS = {
    "EURUSD":{"t":"EURUSD=X","sp":0.3},"USDJPY":{"t":"USDJPY=X","sp":0.3},
    "EURJPY":{"t":"EURJPY=X","sp":0.5},"GBPUSD":{"t":"GBPUSD=X","sp":0.7},
    "GBPJPY":{"t":"GBPJPY=X","sp":0.7},"AUDJPY":{"t":"AUDJPY=X","sp":0.5},
    "NZDJPY":{"t":"NZDJPY=X","sp":0.8},"ZARJPY":{"t":"ZARJPY=X","sp":1.5},
    "CHFJPY":{"t":"CHFJPY=X","sp":1.5},"USDCHF":{"t":"USDCHF=X","sp":1.5},
    "AUDUSD":{"t":"AUDUSD=X","sp":0.5},"EURGBP":{"t":"EURGBP=X","sp":1.0},
    "NZDUSD":{"t":"NZDUSD=X","sp":1.0},"USDCAD":{"t":"USDCAD=X","sp":1.5},
    "CADJPY":{"t":"CADJPY=X","sp":1.5},"AUDCHF":{"t":"AUDCHF=X","sp":2.0},
    "EURAUD":{"t":"EURAUD=X","sp":1.5},"AUDNZD":{"t":"AUDNZD=X","sp":2.0},
    "EURCAD":{"t":"EURCAD=X","sp":2.0},"EURCHF":{"t":"EURCHF=X","sp":1.5},
    "GBPAUD":{"t":"GBPAUD=X","sp":2.0},"AUDCAD":{"t":"AUDCAD=X","sp":2.0},
    "EURNZD":{"t":"EURNZD=X","sp":2.5},"GBPCAD":{"t":"GBPCAD=X","sp":2.5},
    "GBPCHF":{"t":"GBPCHF=X","sp":2.0},"GBPNZD":{"t":"GBPNZD=X","sp":3.0},
}

def pcfg(name,sp):
    j=name.endswith("JPY")
    if j:tv=1000
    elif name in("EURUSD","GBPUSD","AUDUSD","NZDUSD"):tv=1500
    else:tv=1200
    return{"pip":0.01 if j else 0.0001,"pip_mult":100 if j else 10000,"spread":sp,"tick_value":tv}

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

def run_bt(df,cfg,use_fixed_lot=True,zero_spread=True,max_lot=None,disable_martin=False):
    """
    use_fixed_lot=False: リスク%ベース計算
    use_fixed_lot=True : FIXED_LOT (× MARTIN 倍率)
    max_lot=0.1: 最大ロット制限（マーチン含め上限0.1）
    disable_martin=True: マーチン無効化（常にStage1）
    """
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
        sld=atr*SL_MULTS.get(st,1.5);tpd=sld*RR_RATIO;spd=cfg["spread"]*pu
        ep_=row["close"]
        # マーチン倍率（無効化時は常にStage1）
        effective_ms = 0 if disable_martin else min(ms,2)
        mm=MARTIN[effective_ms]

        # ロット計算（固定 or リスク%ベース）
        if use_fixed_lot:
            lots=round(FIXED_LOT*mm,2)  # マーチン倍率適用
        else:
            ra=eq*RISK_PCT/100.0
            sp_pips=sld*pm
            if sp_pips<=0:i+=1;continue
            raw=(ra*mm)/(sp_pips*tv)
            lots=max(0.01,int(raw/0.01)*0.01)

        # 最大ロット上限（案B/C用）
        if max_lot is not None:
            lots=min(lots,max_lot)

        slp=ep_-d*sld;tpp=ep_+d*tpd
        be=False;pc=False;rl=lots;rp=0.0;res="timeout";eb=i;pnl=0.0;to=False
        for j in range(1,MAX_HOLD_BARS+1):
            if i+j>=len(df):break
            b=df.iloc[i+j];ba=b["atr14"] if not np.isnan(b["atr14"]) else atr
            if d==1:
                if not be and(b["high"]-ep_)>=sld*BE_TRIGGER_RR:slp=ep_+spd;be=True
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
                if not be and(ep_-b["low"])>=sld*BE_TRIGGER_RR:slp=ep_-spd;be=True
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

        # スプレッドコスト（zero_spread=Trueで0）
        if not zero_spread:
            pnl-=cfg["spread"]*tv*lots

        if to or res=="BE":cl=0;ms=0
        elif pnl<0:
            cl+=1
            if cl>=3:ms=0;cl=0
            else:ms=min(cl,2)
        else:cl=0;ms=0
        eq+=pnl;ep=max(ep,eq);dd=(ep-eq)/ep*100 if ep>0 else 0;mdd=max(mdd,dd)
        trades.append({"result":res,"pnl":pnl,"lots":lots})
        i=eb+1
    if mseq>0 and cm is not None:monthly.append({"m":cm,"p":(eq-mseq)/mseq*100})
    return trades,mdd,monthly,eq

def summarize(trades,mdd,monthly,final_eq):
    if not trades:return None
    dt=pd.DataFrame(trades);n=len(dt)
    w=dt[dt["pnl"]>0];l=dt[dt["pnl"]<=0]
    wr=len(w)/n*100;tp=dt["pnl"].sum()
    gw=w["pnl"].sum() if len(w)>0 else 0;gl=abs(l["pnl"].sum()) if len(l)>0 else 0
    pf=gw/gl if gl>0 else float("inf")
    mm=pd.DataFrame(monthly)["p"].median() if monthly else 0
    return{"n":n,"wr":wr,"pnl":tp,"pf":pf,"dd":mdd,"mm":mm,"final":final_eq}

def main():
    print("="*100)
    print("BB_Reversal_Martin 固定ロット 0.1 (10,000通貨) 運用バックテスト")
    print(f"初期資金: ¥{INITIAL_EQUITY:,} | 固定LOT: {FIXED_LOT} | FXTF スプレッド0想定")
    print("="*100)

    all_data={}
    for pn,info in FXTF_PAIRS.items():
        cfg=pcfg(pn,info["sp"])
        print(f"  {pn:>8s}... ",end="",flush=True)
        df=fetch(info["t"])
        if df is None or len(df)<300:print("SKIP");continue
        df=calc_ind(df)
        all_data[pn]={"df":df,"cfg":cfg}
        print(f"{len(df)} bars")

    # 5モード比較
    modes=[
        ("現状(変動ロット)",      {"use_fixed_lot":False,"zero_spread":False}),
        ("案A(固定0.1×マーチン)", {"use_fixed_lot":True, "zero_spread":True}),
        ("案B(上限0.1・マーチン有)",{"use_fixed_lot":False,"zero_spread":True,"max_lot":0.1}),
        ("案C(上限0.1・マーチンOFF)",{"use_fixed_lot":False,"zero_spread":True,"max_lot":0.1,"disable_martin":True}),
        ("案D(固定0.1・マーチンOFF)",{"use_fixed_lot":True, "zero_spread":True,"disable_martin":True}),
    ]

    results={}
    for mode_name,kwargs in modes:
        print(f"\n[モード: {mode_name}]")
        for pn,data in all_data.items():
            tr,dd,mo,eq=run_bt(data["df"],data["cfg"],**kwargs)
            s=summarize(tr,dd,mo,eq)
            results.setdefault(pn,{})[mode_name]=s

    # 全モードの純損益比較
    mode_keys=[m[0] for m in modes]
    print(f"\n{'='*120}")
    print(f"  【5モード比較: 純損益 (¥, 3ヶ月)】")
    print(f"{'='*120}")
    header=f"  {'ペア':>7s}"+"".join(f"  {k:>18s}" for k in mode_keys)
    print(header)
    print(f"  {'-'*(7+20*len(mode_keys))}")
    totals={k:0 for k in mode_keys}
    for pn in all_data:
        row=f"  {pn:>7s}"
        for k in mode_keys:
            s=results[pn][k]
            if s:
                v=s["pnl"]
                totals[k]+=v
                row+=f"  {v:>+17,.0f}"
            else:
                row+=f"  {'---':>18s}"
        print(row)
    print(f"  {'-'*(7+20*len(mode_keys))}")
    tr=f"  {'合計':>7s}"
    for k in mode_keys:tr+=f"  {totals[k]:>+17,.0f}"
    print(tr)

    # PF比較
    print(f"\n{'='*120}")
    print(f"  【5モード比較: PF】")
    print(f"{'='*120}")
    print(header)
    print(f"  {'-'*(7+20*len(mode_keys))}")
    for pn in all_data:
        row=f"  {pn:>7s}"
        for k in mode_keys:
            s=results[pn][k]
            if s:
                pfv=f"{s['pf']:.2f}" if s['pf']<100 else "∞"
                row+=f"  {pfv:>18s}"
            else:
                row+=f"  {'---':>18s}"
        print(row)

    rows=[{"pair":pn,**{k:results[pn][k] for k in mode_keys}} for pn in all_data]

    # サマリー
    print(f"\n{'='*120}")
    print(f"  【全体サマリー（全26ペア、3ヶ月）】")
    print(f"{'='*120}")
    print(f"  {'モード':>28s}  {'合計損益(¥)':>14s}  {'月利中央値合計':>12s}  {'プラス収支ペア数':>14s}  {'最大DD平均':>10s}")
    print(f"  {'-'*90}")
    for k in mode_keys:
        vals=[r[k] for r in rows if r[k]]
        if not vals:continue
        total_pnl=sum(v["pnl"] for v in vals)
        mm_sum=sum(v["mm"] for v in vals)
        plus=sum(1 for v in vals if v["pnl"]>0)
        dd_avg=sum(v["dd"] for v in vals)/len(vals)
        print(f"  {k:>28s}  {total_pnl:>+13,.0f}  {mm_sum:>+11.1f}%  {plus:>12d}/26  {dd_avg:>9.1f}%")

    # プラス収支ペアのみ抽出（各モード別）
    print(f"\n{'='*120}")
    print(f"  【プラス収支ペアのみ合算 → 想定月利】")
    print(f"{'='*120}")
    print(f"  {'モード':>28s}  {'ペア数':>6s}  {'3ヶ月損益':>14s}  {'月利合算':>12s}  {'月次絶対額':>14s}")
    print(f"  {'-'*80}")
    for k in mode_keys:
        plus_vals=[r[k] for r in rows if r[k] and r[k]["pnl"]>0]
        if not plus_vals:continue
        pcount=len(plus_vals)
        pnl=sum(v["pnl"] for v in plus_vals)
        mm_sum=sum(v["mm"] for v in plus_vals)
        monthly_abs=INITIAL_EQUITY*mm_sum/100
        print(f"  {k:>28s}  {pcount:>5d}/26  {pnl:>+13,.0f}  {mm_sum:>+11.1f}%  {monthly_abs:>+13,.0f}")

    # 各モードの1年後資産シミュレーション（単利）
    print(f"\n{'='*120}")
    print(f"  【1年後資産シミュレーション (プラスペアのみ、単利)】")
    print(f"{'='*120}")
    print(f"  {'モード':>28s}  {'月利合算':>10s}  {'3ヶ月後':>14s}  {'6ヶ月後':>14s}  {'12ヶ月後':>14s}")
    print(f"  {'-'*92}")
    for k in mode_keys:
        plus_vals=[r[k] for r in rows if r[k] and r[k]["pnl"]>0]
        if not plus_vals:continue
        mm_sum=sum(v["mm"] for v in plus_vals)
        m_abs=INITIAL_EQUITY*mm_sum/100
        e3=INITIAL_EQUITY+m_abs*3
        e6=INITIAL_EQUITY+m_abs*6
        e12=INITIAL_EQUITY+m_abs*12
        print(f"  {k:>28s}  {mm_sum:>+9.1f}%  ¥{e3:>13,.0f}  ¥{e6:>13,.0f}  ¥{e12:>13,.0f}")

    # 推奨案の判定
    print(f"\n{'='*120}")
    print(f"  【判定】")
    print(f"{'='*120}")
    best_k=None; best_pnl=-1e12
    for k in mode_keys:
        tot=sum(r[k]["pnl"] for r in rows if r[k])
        if tot>best_pnl:
            best_pnl=tot; best_k=k
    print(f"  最優秀モード: 【{best_k}】 合計損益 ¥{best_pnl:+,.0f}")

main()
