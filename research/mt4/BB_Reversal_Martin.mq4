//+------------------------------------------------------------------+
//| BB_Reversal_Martin.mq4                                           |
//| BB逆張り＋押し目 × 3段マーチン EA for FXTF(MT4)                  |
//| 15分足専用 / 7通貨ペア同時稼働対応                                |
//+------------------------------------------------------------------+
#property copyright "sokotsudo research"
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| 外部パラメータ                                                    |
//+------------------------------------------------------------------+
input int    MagicNumber       = 10001;   // マジックナンバー（ペア別に変更）
input double RiskPercent       = 0.8;     // 1トレードリスク(%)
input double RR_Ratio          = 2.0;     // リスクリワード比
input double SL_ATR_Mult_BB    = 2.0;     // BB逆張り SL倍率
input double SL_ATR_Mult_FBB   = 1.8;     // 高速BB SL倍率
input double SL_ATR_Mult_PB    = 1.5;     // 押し目 SL倍率
input double ATR_Filter_Mult   = 2.2;     // ATRフィルター倍率
input double BE_Trigger_RR     = 1.0;     // ブレイクイーブン発動RR
input double Partial_Close_RR  = 1.5;     // 部分利確発動RR
input double Partial_Close_Pct = 0.5;     // 部分利確割合(0.5=50%)
input double Trail_ATR_Mult    = 0.5;     // トレーリングSL ATR倍率
input int    MaxHoldingBars    = 20;      // 最大保有バー数
input double Martin1           = 1.0;     // マーチン Stage1 倍率
input double Martin2           = 1.5;     // マーチン Stage2 倍率
input double Martin3           = 2.0;     // マーチン Stage3 倍率
input int    TradingHourStart  = 7;       // 取引開始時刻(UTC)
input int    TradingHourEnd    = 22;      // 取引終了時刻(UTC)
input double MonthlyDDLimit    = 15.0;    // 月次DD停止ライン(%)
input double AnnualDDLimit     = 20.0;    // 年次DD停止ライン(%)
input int    Slippage          = 10;      // スリッページ(points)

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
int    g_entryBar       = 0;
bool   g_beActivated    = false;
bool   g_partialClosed  = false;
double g_slDistance      = 0;
int    g_signalType     = 0;
int    g_currentTicket  = -1;

// 新バー検知用
datetime g_lastBarTime  = 0;

//+------------------------------------------------------------------+
//| サーバー時刻 → UTC変換                                           |
//+------------------------------------------------------------------+
int GetUTCHour()
{
   // FXTFサーバー時刻はGMT+2（夏時間GMT+3）の場合が多い
   // 実際のサーバーオフセットに合わせて調整が必要
   // TimeCurrent()からGMTオフセットを引く
   int gmtOffset = TimeGMTOffset();
   datetime utcTime = TimeCurrent() - gmtOffset;
   MqlDateTime dt;
   TimeToStruct(utcTime, dt);
   return dt.hour;
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
      FileWriteString(handle, IntegerToString(g_entryBar) + "\n");
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
      if(!FileIsEnding(handle)) g_entryBar      = (int)StringToInteger(FileReadString(handle));
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
double CalcLotSize(double slPoints)
{
   if(slPoints <= 0) return 0.01;

   double equity     = AccountEquity();
   double riskAmount = equity * RiskPercent / 100.0;
   double martinMult = GetMartinMultiplier();
   double tickValue  = MarketInfo(Symbol(), MODE_TICKVALUE);
   double tickSize   = MarketInfo(Symbol(), MODE_TICKSIZE);
   double minLot     = MarketInfo(Symbol(), MODE_MINLOT);
   double maxLot     = MarketInfo(Symbol(), MODE_MAXLOT);
   double lotStep    = MarketInfo(Symbol(), MODE_LOTSTEP);

   if(tickValue <= 0 || tickSize <= 0) return minLot;

   double lots = (riskAmount * martinMult) / (slPoints / tickSize * tickValue);

   // ロットステップに合わせて切り捨て
   lots = MathFloor(lots / lotStep) * lotStep;
   lots = MathMax(minLot, MathMin(maxLot, lots));

   return NormalizeDouble(lots, 2);
}

//+------------------------------------------------------------------+
//| SMAトレンド方向判定（5バー前と比較）                              |
//+------------------------------------------------------------------+
bool IsSMAUp(int period, int shift)
{
   double current = iMA(NULL, PERIOD_M15, period, 0, MODE_SMA, PRICE_CLOSE, shift);
   double prev5   = iMA(NULL, PERIOD_M15, period, 0, MODE_SMA, PRICE_CLOSE, shift + 5);
   return current > prev5;
}

//+------------------------------------------------------------------+
//| シグナル判定                                                      |
//+------------------------------------------------------------------+
int CheckSignals()
{
   // shift=1: 確定した前バーの値を使用（リペイント防止）
   double close1    = iClose(NULL, PERIOD_M15, 1);
   double close2    = iClose(NULL, PERIOD_M15, 2);  // 前々バー（BB逆張りの「前足」）
   double rsi       = iRSI(NULL, PERIOD_M15, 10, PRICE_CLOSE, 1);
   double atr       = iATR(NULL, PERIOD_M15, 14, 1);

   // ATR MA(100) を手動計算
   double atrSum = 0;
   for(int k = 1; k <= 100; k++)
      atrSum += iATR(NULL, PERIOD_M15, 14, k);
   double atrMA100 = atrSum / 100.0;

   // --- フィルター ---
   if(atr >= atrMA100 * ATR_Filter_Mult) return SIG_NONE;

   int utcHour = GetUTCHour();
   if(utcHour < TradingHourStart || utcHour >= TradingHourEnd) return SIG_NONE;

   // SMA
   double sma200 = iMA(NULL, PERIOD_M15, 200, 0, MODE_SMA, PRICE_CLOSE, 1);
   double sma50  = iMA(NULL, PERIOD_M15, 50, 0, MODE_SMA, PRICE_CLOSE, 1);
   double sma20  = iMA(NULL, PERIOD_M15, 20, 0, MODE_SMA, PRICE_CLOSE, 1);
   bool sma200Up = IsSMAUp(200, 1);
   bool sma50Up  = IsSMAUp(50, 1);

   // BB(20, 2.5σ)
   double bbUp1  = iBands(NULL, PERIOD_M15, 20, 2.5, 0, PRICE_CLOSE, MODE_UPPER, 1);
   double bbLo1  = iBands(NULL, PERIOD_M15, 20, 2.5, 0, PRICE_CLOSE, MODE_LOWER, 1);
   double bbUp2  = iBands(NULL, PERIOD_M15, 20, 2.5, 0, PRICE_CLOSE, MODE_UPPER, 2);
   double bbLo2  = iBands(NULL, PERIOD_M15, 20, 2.5, 0, PRICE_CLOSE, MODE_LOWER, 2);

   // BB(10, 2.0σ)
   double fbbUp  = iBands(NULL, PERIOD_M15, 10, 2.0, 0, PRICE_CLOSE, MODE_UPPER, 1);
   double fbbLo  = iBands(NULL, PERIOD_M15, 10, 2.0, 0, PRICE_CLOSE, MODE_LOWER, 1);

   // --- シグナル1: BB2.5σ逆張り ---
   // 買い: トレンド上向き + 前々足がBB下限タッチ + 前足がBB下限を回復 + RSI < 38
   if(sma200Up && close2 <= bbLo2 && close1 > bbLo1 && rsi < 38)
      return SIG_BUY_BB;
   // 売り: トレンド下向き + 前々足がBB上限タッチ + 前足がBB上限を下抜け + RSI > 62
   if(!sma200Up && close2 >= bbUp2 && close1 < bbUp1 && rsi > 62)
      return SIG_SELL_BB;

   // --- シグナル2: 高速BB逆張り ---
   // 買い: SMA200・SMA50上向き + BB下限タッチ + RSI < 42
   if(sma200Up && sma50Up && close1 <= fbbLo && rsi < 42)
      return SIG_BUY_FBB;
   // 売り: SMA200・SMA50下向き + BB上限タッチ + RSI > 58
   if(!sma200Up && !sma50Up && close1 >= fbbUp && rsi > 58)
      return SIG_SELL_FBB;

   // --- シグナル3: 押し目・戻り売り ---
   double smaGap = MathAbs(sma20 - sma50);
   if(smaGap >= atr * 2)
   {
      double lowerBand = MathMin(sma20, sma50);
      double upperBand = MathMax(sma20, sma50);

      // 買い: SMA200上向き + 価格がSMA20-50間 + RSI 35-45
      if(sma200Up && close1 >= lowerBand && close1 <= upperBand && rsi >= 35 && rsi <= 45)
         return SIG_BUY_PB;
      // 売り: SMA200下向き + 価格がSMA20-50間 + RSI 55-65
      if(!sma200Up && close1 >= lowerBand && close1 <= upperBand && rsi >= 55 && rsi <= 65)
         return SIG_SELL_PB;
   }

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
   int    barsHeld = Bars - g_entryBar;

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

   // SLをポイントに変換
   double slPoints = slDist / Point;
   double lots = CalcLotSize(slPoints);
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
      g_entryBar = Bars;
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
