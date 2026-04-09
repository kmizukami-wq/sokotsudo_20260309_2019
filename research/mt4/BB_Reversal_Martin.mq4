//+------------------------------------------------------------------+
//| BB_Reversal_Martin.mq4                                           |
//| BB逆張り＋押し目 × 3段マーチン EA for FXTF(MT4)                  |
//| 15分足専用 / 7通貨ペア同時稼働対応                                |
//+------------------------------------------------------------------+
#property copyright "sokotsudo research"
#property version   "1.03"
#property strict

//+------------------------------------------------------------------+
//| 外部パラメータ                                                    |
//+------------------------------------------------------------------+
input int    MagicNumber       = 10001;   // MagicNumber (change per pair)
input double RiskPercent       = 0.8;     // Risk per trade (%)
input double RR_Ratio          = 2.0;     // Risk:Reward ratio
input double SL_ATR_Mult_BB    = 2.0;     // BB reversal SL multiplier
input double SL_ATR_Mult_FBB   = 1.8;     // Fast BB SL multiplier
input double SL_ATR_Mult_PB    = 1.5;     // Pullback SL multiplier
input double ATR_Filter_Mult   = 2.5;     // ATR filter multiplier
input double MaxSpreadPips     = 3.0;     // Max spread (pips) ※超過時エントリー停止
input double BE_Trigger_RR     = 1.0;     // Breakeven trigger RR
input double Partial_Close_RR  = 1.5;     // Partial close trigger RR
input double Partial_Close_Pct = 0.5;     // Partial close ratio (0.5=50%)
input double Trail_ATR_Mult    = 0.5;     // Trailing SL ATR multiplier
input int    MaxHoldingBars    = 20;      // Max holding bars
input double Martin1           = 1.0;     // Martingale Stage1
input double Martin2           = 1.5;     // Martingale Stage2
input double Martin3           = 2.0;     // Martingale Stage3
input int    TradingHourStart  = 9;       // Trading start hour (JST) ※日本時間
input int    TradingHourEnd    = 6;       // Trading end hour (JST) ※Start>Endで日またぎ対応
input double MonthlyDDLimit    = 15.0;    // Monthly DD stop (%)
input double AnnualDDLimit     = 20.0;    // Annual DD stop (%)
input int    Slippage          = 10;      // Slippage (points)

//+------------------------------------------------------------------+
//| シグナルタイプ定数                                                |
//+------------------------------------------------------------------+
#define SIG_NONE     0
#define SIG_BUY_BB   1
#define SIG_SELL_BB  -1
#define SIG_BUY_FBB  2
#define SIG_SELL_FBB -2
#define SIG_BUY_PB   3
#define SIG_SELL_PB  -3

//+------------------------------------------------------------------+
//| グローバル変数                                                    |
//+------------------------------------------------------------------+
int    g_martinStage    = 0;
int    g_consecLosses   = 0;
double g_monthStartEq   = 0;
double g_yearStartEq    = 0;
int    g_currentMonth   = 0;
int    g_currentYear    = 0;
bool   g_systemStopped  = false;
string g_stopReason     = "";

// ポジション管理用
datetime g_entryTime     = 0;
bool   g_beActivated    = false;
bool   g_partialClosed  = false;
double g_slDistance      = 0;
int    g_signalType     = 0;
int    g_currentTicket  = -1;

// 新バー検知用
datetime g_lastBarTime  = 0;

//+------------------------------------------------------------------+
//| 時刻関連ユーティリティ                                            |
//+------------------------------------------------------------------+
input int    ServerGMTOffset   = 2;       // Server GMT offset (FXTF MT4=2, 夏時間=3)

int GetJSTHour()
{
   // サーバー時刻 → JST(UTC+9) に変換
   // 例: ServerGMTOffset=2 → JST = サーバー時刻 + 7時間
   int jstOffset = 9 - ServerGMTOffset;
   datetime jstTime = TimeCurrent() + jstOffset * 3600;
   return TimeHour(jstTime);
}

int GetUTCHour()
{
   // ログ用: サーバー時刻 → UTC変換
   datetime utcTime = TimeCurrent() - ServerGMTOffset * 3600;
   return TimeHour(utcTime);
}

//+------------------------------------------------------------------+
//| 新バー検知                                                        |
//+------------------------------------------------------------------+
bool IsNewBar()
{
   datetime currentBarTime = iTime(NULL, PERIOD_M15, 0);
   if(currentBarTime != g_lastBarTime)
   {
      g_lastBarTime = currentBarTime;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| 状態保存                                                          |
//+------------------------------------------------------------------+
void SaveState()
{
   string filename = "BB_Martin_" + Symbol() + "_" + IntegerToString(MagicNumber) + ".dat";
   int handle = FileOpen(filename, FILE_WRITE|FILE_TXT);
   if(handle != INVALID_HANDLE)
   {
      FileWriteString(handle, IntegerToString(g_martinStage) + "\n");
      FileWriteString(handle, IntegerToString(g_consecLosses) + "\n");
      FileWriteString(handle, DoubleToString(g_monthStartEq, 2) + "\n");
      FileWriteString(handle, DoubleToString(g_yearStartEq, 2) + "\n");
      FileWriteString(handle, IntegerToString(g_currentMonth) + "\n");
      FileWriteString(handle, IntegerToString(g_currentYear) + "\n");
      FileWriteString(handle, IntegerToString(g_systemStopped) + "\n");
      // ポジション管理状態
      FileWriteString(handle, IntegerToString((int)g_entryTime) + "\n");
      FileWriteString(handle, IntegerToString(g_beActivated) + "\n");
      FileWriteString(handle, IntegerToString(g_partialClosed) + "\n");
      FileWriteString(handle, DoubleToString(g_slDistance, 6) + "\n");
      FileWriteString(handle, IntegerToString(g_signalType) + "\n");
      FileClose(handle);
   }
}

//+------------------------------------------------------------------+
//| 状態読み込み                                                      |
//+------------------------------------------------------------------+
void LoadState()
{
   string filename = "BB_Martin_" + Symbol() + "_" + IntegerToString(MagicNumber) + ".dat";
   if(!FileIsExist(filename))
      return;

   int handle = FileOpen(filename, FILE_READ|FILE_TXT);
   if(handle != INVALID_HANDLE)
   {
      if(!FileIsEnding(handle)) g_martinStage   = (int)StringToInteger(FileReadString(handle));
      if(!FileIsEnding(handle)) g_consecLosses  = (int)StringToInteger(FileReadString(handle));
      if(!FileIsEnding(handle)) g_monthStartEq  = StringToDouble(FileReadString(handle));
      if(!FileIsEnding(handle)) g_yearStartEq   = StringToDouble(FileReadString(handle));
      if(!FileIsEnding(handle)) g_currentMonth  = (int)StringToInteger(FileReadString(handle));
      if(!FileIsEnding(handle)) g_currentYear   = (int)StringToInteger(FileReadString(handle));
      if(!FileIsEnding(handle)) g_systemStopped = (bool)StringToInteger(FileReadString(handle));
      if(!FileIsEnding(handle)) g_entryTime     = (datetime)StringToInteger(FileReadString(handle));
      if(!FileIsEnding(handle)) g_beActivated   = (bool)StringToInteger(FileReadString(handle));
      if(!FileIsEnding(handle)) g_partialClosed = (bool)StringToInteger(FileReadString(handle));
      if(!FileIsEnding(handle)) g_slDistance     = StringToDouble(FileReadString(handle));
      if(!FileIsEnding(handle)) g_signalType    = (int)StringToInteger(FileReadString(handle));
      FileClose(handle);
      Print("State loaded: martin=", g_martinStage, " losses=", g_consecLosses);
   }
}

//+------------------------------------------------------------------+
//| 自分のポジションを探す                                            |
//+------------------------------------------------------------------+
int FindMyOrder()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         continue;
      if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
         return OrderTicket();
   }
   return -1;
}

//+------------------------------------------------------------------+
//| マーチン倍率取得                                                  |
//+------------------------------------------------------------------+
double GetMartinMultiplier()
{
   switch(g_martinStage)
   {
      case 0:  return Martin1;
      case 1:  return Martin2;
      case 2:  return Martin3;
      default: return Martin1;
   }
}

//+------------------------------------------------------------------+
//| ロットサイズ計算                                                  |
//+------------------------------------------------------------------+
double CalcLotSize(double slDist)
{
   if(slDist <= 0) return 0.01;

   double equity     = AccountEquity();
   double riskAmount = equity * RiskPercent / 100.0;
   double martinMult = GetMartinMultiplier();
   double tickValue  = MarketInfo(Symbol(), MODE_TICKVALUE);
   double tickSize   = MarketInfo(Symbol(), MODE_TICKSIZE);
   double minLot     = MarketInfo(Symbol(), MODE_MINLOT);
   double maxLot     = MarketInfo(Symbol(), MODE_MAXLOT);
   double lotStep    = MarketInfo(Symbol(), MODE_LOTSTEP);

   if(tickValue <= 0 || tickSize <= 0) return minLot;

   // SL金額/lot = slDist / tickSize * tickValue
   double slCostPerLot = slDist / tickSize * tickValue;
   double lots = (riskAmount * martinMult) / slCostPerLot;

   Print("[DEBUG] LotCalc: equity=", equity, " risk=", riskAmount, " martin=", martinMult,
         " slDist=", DoubleToString(slDist, 5), " slCost/lot=", DoubleToString(slCostPerLot, 2),
         " raw_lots=", DoubleToString(lots, 4));

   // ロットステップに合わせて切り捨て
   lots = MathFloor(lots / lotStep) * lotStep;
   lots = MathMax(minLot, MathMin(maxLot, lots));

   return NormalizeDouble(lots, 2);
}

//+------------------------------------------------------------------+
//| シグナル判定（最適化版）                                          |
//+------------------------------------------------------------------+
int CheckSignals()
{
   // --- 軽量フィルター（先に処理して早期リターン） ---

   // 1. 時間フィルター（計算不要で最速）
   int jstHour = GetJSTHour();
   bool inWindow;
   if(TradingHourStart <= TradingHourEnd)
      inWindow = (jstHour >= TradingHourStart && jstHour < TradingHourEnd);
   else
      inWindow = (jstHour >= TradingHourStart || jstHour < TradingHourEnd);

   if(!inWindow)
   {
      Print("[DEBUG] Filtered: JST hour=", jstHour, " outside ", TradingHourStart, "-", TradingHourEnd);
      return SIG_NONE;
   }

   // 2. スプレッドフィルター（FXTF早朝・指標時対策）
   double spreadPips = MarketInfo(Symbol(), MODE_SPREAD) * Point / (Point * MathPow(10, Digits % 2));
   if(Digits == 3 || Digits == 5)
      spreadPips = MarketInfo(Symbol(), MODE_SPREAD) / 10.0;
   else
      spreadPips = MarketInfo(Symbol(), MODE_SPREAD);

   if(spreadPips > MaxSpreadPips)
   {
      Print("[DEBUG] Filtered: Spread=", DoubleToString(spreadPips,1), " pips > MaxSpread=", MaxSpreadPips);
      return SIG_NONE;
   }

   // 3. ATRフィルター（100回ループ → ATR(100)の1回呼び出しに高速化）
   double atr     = iATR(NULL, PERIOD_M15, 14, 1);
   double atrLong = iATR(NULL, PERIOD_M15, 100, 1);
   if(atr >= atrLong * ATR_Filter_Mult)
   {
      Print("[DEBUG] Filtered: ATR(14)=", DoubleToString(atr,5), " >= ATR(100)*", ATR_Filter_Mult,
            " (", DoubleToString(atrLong * ATR_Filter_Mult, 5), ")");
      return SIG_NONE;
   }

   // --- 価格・インジケータ取得 ---
   double close1 = iClose(NULL, PERIOD_M15, 1);
   double close2 = iClose(NULL, PERIOD_M15, 2);
   double open1  = iOpen(NULL, PERIOD_M15, 1);   // ローソク足方向確認用
   double rsi    = iRSI(NULL, PERIOD_M15, 10, PRICE_CLOSE, 1);

   // SMA（1回ずつ取得、IsSMAUp統合で重複呼び出し排除）
   double sma200    = iMA(NULL, PERIOD_M15, 200, 0, MODE_SMA, PRICE_CLOSE, 1);
   double sma200_p5 = iMA(NULL, PERIOD_M15, 200, 0, MODE_SMA, PRICE_CLOSE, 6);
   double sma50     = iMA(NULL, PERIOD_M15, 50, 0, MODE_SMA, PRICE_CLOSE, 1);
   double sma50_p5  = iMA(NULL, PERIOD_M15, 50, 0, MODE_SMA, PRICE_CLOSE, 6);
   bool sma200Up = (sma200 > sma200_p5);
   bool sma50Up  = (sma50 > sma50_p5);

   Print("[DEBUG] Filters passed: JST=", jstHour, " Spread=", DoubleToString(spreadPips,1),
         " ATR=", DoubleToString(atr,5), " Close=", DoubleToString(close1,5),
         " RSI=", DoubleToString(rsi,1), " SMA200up=", sma200Up);

   // --- シグナル1: BB2.5σ逆張り（ローソク足方向確認追加） ---
   double bbUp1 = iBands(NULL, PERIOD_M15, 20, 2.5, 0, PRICE_CLOSE, MODE_UPPER, 1);
   double bbLo1 = iBands(NULL, PERIOD_M15, 20, 2.5, 0, PRICE_CLOSE, MODE_LOWER, 1);
   double bbUp2 = iBands(NULL, PERIOD_M15, 20, 2.5, 0, PRICE_CLOSE, MODE_UPPER, 2);
   double bbLo2 = iBands(NULL, PERIOD_M15, 20, 2.5, 0, PRICE_CLOSE, MODE_LOWER, 2);

   // BUY: 前々足がBB下限突破 → 前足で戻り＋陽線確認
   if(sma200Up && close2 <= bbLo2 && close1 > bbLo1 && close1 > open1 && rsi < 42)
   { Print("[SIGNAL] BUY BB_reversal: close2=", close2, "<=bbLo2=", bbLo2, " close1=", close1, ">bbLo1=", bbLo1, " bullish candle, RSI=", rsi); return SIG_BUY_BB; }
   // SELL: 前々足がBB上限突破 → 前足で戻り＋陰線確認
   if(!sma200Up && close2 >= bbUp2 && close1 < bbUp1 && close1 < open1 && rsi > 58)
   { Print("[SIGNAL] SELL BB_reversal: close2=", close2, ">=bbUp2=", bbUp2, " close1=", close1, "<bbUp1=", bbUp1, " bearish candle, RSI=", rsi); return SIG_SELL_BB; }

   // --- シグナル2: 高速BB逆張り ---
   double fbbUp = iBands(NULL, PERIOD_M15, 10, 2.0, 0, PRICE_CLOSE, MODE_UPPER, 1);
   double fbbLo = iBands(NULL, PERIOD_M15, 10, 2.0, 0, PRICE_CLOSE, MODE_LOWER, 1);

   if(sma200Up && sma50Up && close1 <= fbbLo && rsi < 48)
   { Print("[SIGNAL] BUY Fast_BB: close1=", close1, "<=fbbLo=", fbbLo, " RSI=", rsi); return SIG_BUY_FBB; }
   if(!sma200Up && !sma50Up && close1 >= fbbUp && rsi > 52)
   { Print("[SIGNAL] SELL Fast_BB: close1=", close1, ">=fbbUp=", fbbUp, " RSI=", rsi); return SIG_SELL_FBB; }

   // --- シグナル3: 押し目・戻り売り ---
   double sma20 = iMA(NULL, PERIOD_M15, 20, 0, MODE_SMA, PRICE_CLOSE, 1);
   double smaGap = MathAbs(sma20 - sma50);
   if(smaGap >= atr * 2)
   {
      double lowerBand = MathMin(sma20, sma50);
      double upperBand = MathMax(sma20, sma50);

      if(sma200Up && close1 >= lowerBand && close1 <= upperBand && rsi >= 30 && rsi <= 50)
      { Print("[SIGNAL] BUY Pullback: close1=", close1, " in [", lowerBand, ",", upperBand, "] RSI=", rsi); return SIG_BUY_PB; }
      if(!sma200Up && close1 >= lowerBand && close1 <= upperBand && rsi >= 50 && rsi <= 70)
      { Print("[SIGNAL] SELL Pullback: close1=", close1, " in [", lowerBand, ",", upperBand, "] RSI=", rsi); return SIG_SELL_PB; }
   }

   Print("[DEBUG] No signal: SMA200up=", sma200Up, " SMA50up=", sma50Up,
         " BBlo=", DoubleToString(bbLo1,5), " BBhi=", DoubleToString(bbUp1,5),
         " FBBlo=", DoubleToString(fbbLo,5), " FBBhi=", DoubleToString(fbbUp,5),
         " SMAGap=", DoubleToString(smaGap,5), " ATR*2=", DoubleToString(atr*2,5));
   return SIG_NONE;
}

//+------------------------------------------------------------------+
//| SL倍率取得（シグナルタイプ別）                                    |
//+------------------------------------------------------------------+
double GetSLMultiplier(int sigType)
{
   int absSig = MathAbs(sigType);
   if(absSig == 1) return SL_ATR_Mult_BB;   // BB逆張り
   if(absSig == 2) return SL_ATR_Mult_FBB;  // 高速BB
   if(absSig == 3) return SL_ATR_Mult_PB;   // 押し目
   return SL_ATR_Mult_PB;
}

//+------------------------------------------------------------------+
//| ポジション管理（BE/部分利確/トレーリング/時間切れ）                |
//+------------------------------------------------------------------+
void ManagePosition()
{
   int ticket = FindMyOrder();
   if(ticket < 0) return;

   if(!OrderSelect(ticket, SELECT_BY_TICKET))
      return;

   double entry  = OrderOpenPrice();
   double curSL  = OrderStopLoss();
   double curTP  = OrderTakeProfit();
   int    type   = OrderType();
   double lots   = OrderLots();
   double atr    = iATR(NULL, PERIOD_M15, 14, 1);
   int    barsHeld = (g_entryTime > 0) ? iBarShift(NULL, PERIOD_M15, g_entryTime, false) : 0;

   if(type == OP_BUY)
   {
      double unrealized = Bid - entry;

      // ブレイクイーブン
      if(!g_beActivated && unrealized >= g_slDistance * BE_Trigger_RR)
      {
         double newSL = entry + MarketInfo(Symbol(), MODE_SPREAD) * Point;
         newSL = NormalizeDouble(newSL, Digits);
         if(newSL > curSL)
         {
            if(OrderModify(ticket, entry, newSL, curTP, 0, clrBlue))
            {
               g_beActivated = true;
               Print("BE activated for ticket #", ticket);
            }
         }
      }

      // 部分利確
      if(!g_partialClosed && unrealized >= g_slDistance * Partial_Close_RR)
      {
         double closeLots = NormalizeDouble(lots * Partial_Close_Pct, 2);
         double minLot = MarketInfo(Symbol(), MODE_MINLOT);
         if(closeLots >= minLot && (lots - closeLots) >= minLot)
         {
            if(OrderClose(ticket, closeLots, Bid, Slippage, clrGreen))
            {
               g_partialClosed = true;
               Print("Partial close: ", closeLots, " lots at ", Bid);

               // 残りポジションのチケットを再取得
               g_currentTicket = FindMyOrder();
               if(g_currentTicket > 0 && OrderSelect(g_currentTicket, SELECT_BY_TICKET))
               {
                  // トレーリングSL開始
                  double trailSL = NormalizeDouble(Bid - atr * Trail_ATR_Mult, Digits);
                  if(trailSL > OrderStopLoss())
                     OrderModify(g_currentTicket, OrderOpenPrice(), trailSL, OrderTakeProfit(), 0, clrBlue);
               }
               SaveState();
               return;
            }
         }
      }

      // トレーリング更新（部分利確後）
      if(g_partialClosed)
      {
         double trailSL = NormalizeDouble(Bid - atr * Trail_ATR_Mult, Digits);
         if(trailSL > curSL)
         {
            OrderModify(ticket, entry, trailSL, curTP, 0, clrBlue);
         }
      }

      // 最大保有期間チェック
      if(barsHeld >= MaxHoldingBars)
      {
         OrderClose(ticket, lots, Bid, Slippage, clrRed);
         Print("Time exit (BUY) after ", barsHeld, " bars");
         OnTradeClose(0);  // pnl=0としてマーチン判定させない（TIME決済）
         return;
      }
   }
   else if(type == OP_SELL)
   {
      double unrealized = entry - Ask;

      if(!g_beActivated && unrealized >= g_slDistance * BE_Trigger_RR)
      {
         double newSL = entry - MarketInfo(Symbol(), MODE_SPREAD) * Point;
         newSL = NormalizeDouble(newSL, Digits);
         if(newSL < curSL || curSL == 0)
         {
            if(OrderModify(ticket, entry, newSL, curTP, 0, clrRed))
            {
               g_beActivated = true;
               Print("BE activated for ticket #", ticket);
            }
         }
      }

      if(!g_partialClosed && unrealized >= g_slDistance * Partial_Close_RR)
      {
         double closeLots = NormalizeDouble(lots * Partial_Close_Pct, 2);
         double minLot = MarketInfo(Symbol(), MODE_MINLOT);
         if(closeLots >= minLot && (lots - closeLots) >= minLot)
         {
            if(OrderClose(ticket, closeLots, Ask, Slippage, clrGreen))
            {
               g_partialClosed = true;
               Print("Partial close: ", closeLots, " lots at ", Ask);
               g_currentTicket = FindMyOrder();
               if(g_currentTicket > 0 && OrderSelect(g_currentTicket, SELECT_BY_TICKET))
               {
                  double trailSL = NormalizeDouble(Ask + atr * Trail_ATR_Mult, Digits);
                  if(trailSL < OrderStopLoss() || OrderStopLoss() == 0)
                     OrderModify(g_currentTicket, OrderOpenPrice(), trailSL, OrderTakeProfit(), 0, clrRed);
               }
               SaveState();
               return;
            }
         }
      }

      if(g_partialClosed)
      {
         double trailSL = NormalizeDouble(Ask + atr * Trail_ATR_Mult, Digits);
         if(trailSL < curSL || curSL == 0)
         {
            OrderModify(ticket, entry, trailSL, curTP, 0, clrRed);
         }
      }

      if(barsHeld >= MaxHoldingBars)
      {
         OrderClose(ticket, lots, Ask, Slippage, clrRed);
         Print("Time exit (SELL) after ", barsHeld, " bars");
         OnTradeClose(0);
         return;
      }
   }
}

//+------------------------------------------------------------------+
//| トレード決済後のマーチン処理                                      |
//+------------------------------------------------------------------+
void OnTradeClose(double profit)
{
   if(profit < 0)
   {
      // SL損失 → マーチン段階アップ
      g_consecLosses++;
      if(g_consecLosses >= 3)
      {
         g_martinStage = 0;
         g_consecLosses = 0;
      }
      else
      {
         g_martinStage = MathMin(g_consecLosses, 2);
      }
   }
   else
   {
      // 利益 or BE or TIME → リセット
      g_consecLosses = 0;
      g_martinStage = 0;
   }

   // ポジション管理フラグリセット
   g_beActivated = false;
   g_partialClosed = false;
   g_slDistance = 0;
   g_signalType = 0;
   g_currentTicket = -1;

   SaveState();
}

//+------------------------------------------------------------------+
//| 直前の決済を検知してマーチン更新                                  |
//+------------------------------------------------------------------+
void CheckClosedOrders()
{
   // ポジションが消えていたら決済されたと判断
   if(g_currentTicket > 0 && FindMyOrder() < 0)
   {
      // 決済されたオーダーの損益を取得
      if(OrderSelect(g_currentTicket, SELECT_BY_TICKET, MODE_HISTORY))
      {
         double profit = OrderProfit() + OrderSwap() + OrderCommission();
         Print("Order #", g_currentTicket, " closed. Profit=", profit);
         OnTradeClose(profit);
      }
      else
      {
         OnTradeClose(0);
      }
   }
}

//+------------------------------------------------------------------+
//| DD停止チェック                                                    |
//+------------------------------------------------------------------+
void CheckDDStop()
{
   double equity = AccountEquity();

   // 月替わり検知
   if(Month() != g_currentMonth)
   {
      g_currentMonth = Month();
      g_monthStartEq = equity;
      // 月替わりで停止解除
      if(g_systemStopped && StringFind(g_stopReason, "月次") >= 0)
      {
         g_systemStopped = false;
         g_stopReason = "";
         Print("月替わりにより停止解除");
      }
   }
   // 年替わり検知
   if(Year() != g_currentYear)
   {
      g_currentYear = Year();
      g_yearStartEq = equity;
      if(g_systemStopped)
      {
         g_systemStopped = false;
         g_stopReason = "";
         Print("年替わりにより停止解除");
      }
   }

   if(g_monthStartEq > 0)
   {
      double monthlyDD = (equity - g_monthStartEq) / g_monthStartEq * 100.0;
      if(monthlyDD <= -MonthlyDDLimit)
      {
         g_systemStopped = true;
         g_stopReason = StringFormat("月次DD停止 (%.1f%%)", monthlyDD);
         Print(g_stopReason);
      }
   }
   if(g_yearStartEq > 0)
   {
      double annualDD = (equity - g_yearStartEq) / g_yearStartEq * 100.0;
      if(annualDD <= -AnnualDDLimit)
      {
         g_systemStopped = true;
         g_stopReason = StringFormat("年次DD停止 (%.1f%%)", annualDD);
         Print(g_stopReason);
      }
   }
}

//+------------------------------------------------------------------+
//| 新規エントリー                                                    |
//+------------------------------------------------------------------+
void TryEntry()
{
   if(FindMyOrder() >= 0) return;  // 既にポジションあり

   int signal = CheckSignals();
   if(signal == SIG_NONE) return;

   double atr = iATR(NULL, PERIOD_M15, 14, 1);
   double slMult = GetSLMultiplier(signal);
   double slDist = atr * slMult;
   double tpDist = slDist * RR_Ratio;

   double lots = CalcLotSize(slDist);
   if(lots <= 0) return;

   double entryPrice, slPrice, tpPrice;
   int cmd;
   color arrowColor;

   if(signal > 0)  // BUY
   {
      cmd = OP_BUY;
      entryPrice = Ask;
      slPrice = NormalizeDouble(entryPrice - slDist, Digits);
      tpPrice = NormalizeDouble(entryPrice + tpDist, Digits);
      arrowColor = clrBlue;
   }
   else  // SELL
   {
      cmd = OP_SELL;
      entryPrice = Bid;
      slPrice = NormalizeDouble(entryPrice + slDist, Digits);
      tpPrice = NormalizeDouble(entryPrice - tpDist, Digits);
      arrowColor = clrRed;
   }

   string comment = StringFormat("BB_M_%d_S%d", MathAbs(signal), g_martinStage + 1);

   int ticket = OrderSend(Symbol(), cmd, lots, entryPrice, Slippage,
                           slPrice, tpPrice, comment, MagicNumber, 0, arrowColor);

   if(ticket > 0)
   {
      g_currentTicket = ticket;
      g_entryTime = TimeCurrent();
      g_beActivated = false;
      g_partialClosed = false;
      g_slDistance = slDist;
      g_signalType = signal;
      Print(StringFormat("Entry: %s %s %.2f lots @ %.5f SL=%.5f TP=%.5f Stage=%d",
            Symbol(), (cmd == OP_BUY ? "BUY" : "SELL"), lots, entryPrice, slPrice, tpPrice,
            g_martinStage + 1));
      SaveState();
   }
   else
   {
      Print("OrderSend failed: ", GetLastError());
   }
}

//+------------------------------------------------------------------+
//| チャートコメント表示                                              |
//+------------------------------------------------------------------+
void ShowStatus()
{
   string status = StringFormat(
      "BB Reversal Martin EA\n"
      "Symbol: %s | Magic: %d\n"
      "Martin Stage: %d (x%.1f) | Consec Losses: %d\n"
      "Equity: %.0f | Month Start: %.0f\n"
      "Monthly PnL: %+.1f%%\n"
      "System: %s\n"
      "Position: %s | BE: %s | Partial: %s",
      Symbol(), MagicNumber,
      g_martinStage + 1, GetMartinMultiplier(), g_consecLosses,
      AccountEquity(), g_monthStartEq,
      g_monthStartEq > 0 ? (AccountEquity() - g_monthStartEq) / g_monthStartEq * 100.0 : 0.0,
      g_systemStopped ? "STOPPED (" + g_stopReason + ")" : "ACTIVE",
      FindMyOrder() >= 0 ? "OPEN" : "NONE",
      g_beActivated ? "YES" : "NO",
      g_partialClosed ? "YES" : "NO"
   );
   Comment(status);
}

//+------------------------------------------------------------------+
//| 初期化                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   // チャートが15分足でなければ警告
   if(Period() != PERIOD_M15)
      Alert("Warning: This EA is designed for M15 timeframe!");

   g_currentMonth = Month();
   g_currentYear  = Year();
   g_monthStartEq = AccountEquity();
   g_yearStartEq  = AccountEquity();
   g_lastBarTime  = iTime(NULL, PERIOD_M15, 0);

   // 状態復元
   LoadState();

   // 既存ポジションの確認
   g_currentTicket = FindMyOrder();

   Print(StringFormat("BB_Reversal_Martin EA initialized on %s M15 (Magic=%d)", Symbol(), MagicNumber));
   Print(StringFormat("Martin: [%.1f, %.1f, %.1f] | Risk: %.1f%% | RR: %.1f",
         Martin1, Martin2, Martin3, RiskPercent, RR_Ratio));

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| メインティック処理                                                |
//+------------------------------------------------------------------+
void OnTick()
{
   // 新しい15分バーでのみ処理
   if(!IsNewBar())
   {
      // ティックベースのポジション管理のみ実行
      if(FindMyOrder() >= 0)
         ManagePosition();
      return;
   }

   Print("[DEBUG] === NewBar on ", Symbol(), " === ServerTime=", TimeToString(TimeCurrent()),
         " JST=", GetJSTHour(), " UTC=", GetUTCHour(), " ServerGMT+", ServerGMTOffset);

   // DD停止チェック
   CheckDDStop();

   // 決済検知 → マーチン更新
   CheckClosedOrders();

   // ステータス表示
   ShowStatus();

   if(g_systemStopped)
      return;

   // ポジション管理
   if(FindMyOrder() >= 0)
   {
      ManagePosition();
      return;
   }

   // 新規エントリー
   TryEntry();
}

//+------------------------------------------------------------------+
//| 終了処理                                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   SaveState();
   Comment("");
   Print("BB_Reversal_Martin EA deinitialized. Reason=", reason);
}
//+------------------------------------------------------------------+
