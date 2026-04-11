//+------------------------------------------------------------------+
//| ZscoreScalper.mq4 - 15分足Zスコア逆張りスキャルパー              |
//| 8ペア対応 / FXTF 1万通貨 / スプレッドゼロ前提                    |
//+------------------------------------------------------------------+
#property copyright "sokotsudo"
#property version   "1.00"
#property strict

//--- 入力パラメータ
input int    Window      = 30;     // Zscore Window (M15 x 30 = 7.5h)
input double EntryZ      = 0.51;   // Entry Z threshold
input double ExitZ       = 0.5;    // Exit Z threshold
input double StopZ       = 6.0;    // Stop Z threshold
input double TimeoutH    = 2.0;    // Timeout (hours)
input int    TPpips      = 15;     // Take Profit (pips) 0=off
input int    SLpips      = 15;     // Stop Loss (pips) 0=off
input double LotSize     = 0.1;    // Lot Size (0.1=10000, 0.01=1000)
input int    MagicNumber = 20260411; // Magic Number
input int    Slippage    = 3;      // Slippage (points)
input int    TradeStartH = 0;      // Trade Start Hour (UTC)
input int    TradeEndH   = 21;     // Trade End Hour (UTC)
input int    BrokerUTCOffset = 3;  // Broker UTC Offset (FXTF summer=3, winter=2)

//--- グローバル変数
double g_mean[];
double g_std[];
double g_close[];
int    g_size;

//+------------------------------------------------------------------+
//| 取引時間帯チェック（UTC基準）                                       |
//+------------------------------------------------------------------+
bool IsTradeTime()
{
   // ブローカー時刻からUTC時刻を算出
   int brokerHour = TimeHour(TimeCurrent());
   int utcHour = brokerHour - BrokerUTCOffset;
   if(utcHour < 0) utcHour += 24;
   if(utcHour >= 24) utcHour -= 24;

   // UTC 0:00 ~ 21:00 が取引時間
   if(TradeStartH <= TradeEndH)
      return(utcHour >= TradeStartH && utcHour < TradeEndH);
   else // 日跨ぎ（例: 22~5）
      return(utcHour >= TradeStartH || utcHour < TradeEndH);
}

//+------------------------------------------------------------------+
//| 取引時間外のポジション強制決済                                       |
//+------------------------------------------------------------------+
void CloseIfOutsideTradeTime()
{
   if(IsTradeTime()) return;

   int pos = GetCurrentPosition();
   if(pos != 0)
   {
      int brokerHour = TimeHour(TimeCurrent());
      int utcHour = brokerHour - BrokerUTCOffset;
      if(utcHour < 0) utcHour += 24;

      Print("取引時間外決済: ", Symbol(), " UTC=", utcHour, ":00");
      ClosePosition();
   }
}

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   if(Period() != PERIOD_M15)
   {
      Alert("このEAは15分足専用です。チャートを15分足に変更してください。");
      return(INIT_FAILED);
   }

   Print("ZscoreScalper 初期化完了: ", Symbol(),
         " W=", Window, " Entry=", EntryZ, " Exit=", ExitZ,
         " Stop=", StopZ, " TO=", TimeoutH, "h Lot=", LotSize,
         " 取引時間=UTC", TradeStartH, "-", TradeEndH,
         " BrokerOffset=UTC+", BrokerUTCOffset);

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                    |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("ZscoreScalper 停止: ", Symbol(), " 理由=", reason);
}

//+------------------------------------------------------------------+
//| Zスコア計算                                                        |
//+------------------------------------------------------------------+
double CalcZscore()
{
   if(Bars < Window + 1) return(0);

   // 直近Window本の終値から平均と標準偏差を計算
   double sum = 0;
   double sum2 = 0;

   for(int i = 0; i < Window; i++)
   {
      double c = iClose(Symbol(), PERIOD_M15, i);
      sum += c;
      sum2 += c * c;
   }

   double mean = sum / Window;
   double variance = (sum2 / Window) - (mean * mean);

   if(variance <= 0) return(0);

   double std = MathSqrt(variance);
   double current_price = iClose(Symbol(), PERIOD_M15, 0);
   double zscore = (current_price - mean) / std;

   return(zscore);
}

//+------------------------------------------------------------------+
//| 現在のポジションを取得                                              |
//+------------------------------------------------------------------+
int GetCurrentPosition()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
      {
         if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
         {
            if(OrderType() == OP_BUY) return(1);
            if(OrderType() == OP_SELL) return(-1);
         }
      }
   }
   return(0);
}

//+------------------------------------------------------------------+
//| ポジションのエントリー時刻を取得                                    |
//+------------------------------------------------------------------+
datetime GetEntryTime()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
      {
         if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
         {
            return(OrderOpenTime());
         }
      }
   }
   return(0);
}

//+------------------------------------------------------------------+
//| ポジションのエントリー価格を取得                                     |
//+------------------------------------------------------------------+
double GetEntryPrice()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
      {
         if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
         {
            return(OrderOpenPrice());
         }
      }
   }
   return(0);
}

//+------------------------------------------------------------------+
//| ポジション決済                                                      |
//+------------------------------------------------------------------+
bool ClosePosition()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
      {
         if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
         {
            double price;
            if(OrderType() == OP_BUY)
               price = MarketInfo(Symbol(), MODE_BID);
            else
               price = MarketInfo(Symbol(), MODE_ASK);

            bool result = OrderClose(OrderTicket(), OrderLots(), price, Slippage, CLR_NONE);
            if(!result)
               Print("決済エラー: ", GetLastError(), " Ticket=", OrderTicket());
            return(result);
         }
      }
   }
   return(false);
}

//+------------------------------------------------------------------+
//| 新規エントリー                                                      |
//+------------------------------------------------------------------+
bool OpenPosition(int direction)
{
   double price;
   int cmd;
   color clr;
   string comment;

   if(direction == 1) // ロング
   {
      price = MarketInfo(Symbol(), MODE_ASK);
      cmd = OP_BUY;
      clr = clrBlue;
      comment = "Zscore_Long";
   }
   else if(direction == -1) // ショート
   {
      price = MarketInfo(Symbol(), MODE_BID);
      cmd = OP_SELL;
      clr = clrRed;
      comment = "Zscore_Short";
   }
   else return(false);

   int ticket = OrderSend(Symbol(), cmd, LotSize, price, Slippage,
                           0, 0, comment, MagicNumber, 0, clr);

   if(ticket < 0)
   {
      Print("エントリーエラー: ", GetLastError(),
            " ", Symbol(), " ", (direction==1?"BUY":"SELL"), " @", price);
      return(false);
   }

   Print("エントリー: ", Symbol(), " ", (direction==1?"BUY":"SELL"),
         " @", price, " Ticket=", ticket);
   return(true);
}

//+------------------------------------------------------------------+
//| Expert tick function                                                |
//+------------------------------------------------------------------+
void OnTick()
{
   // 取引時間外はポジション決済して終了
   CloseIfOutsideTradeTime();

   int pos = GetCurrentPosition();

   //=== TP/SL判定: 毎ティック ===
   if(pos != 0 && (TPpips > 0 || SLpips > 0))
   {
      double entryPrice = GetEntryPrice();
      double currentPrice = MarketInfo(Symbol(), MODE_BID);
      double pipSize = MarketInfo(Symbol(), MODE_POINT) * 10; // 1pip

      double unrealizedPips = (currentPrice - entryPrice) * pos / pipSize;

      // TP利確
      if(TPpips > 0 && unrealizedPips >= TPpips)
      {
         Print("TP利確: ", Symbol(), " +", DoubleToStr(unrealizedPips, 1),
               "pips @", DoubleToStr(currentPrice, (int)MarketInfo(Symbol(), MODE_DIGITS)));
         ClosePosition();
         return;
      }

      // SL損切り
      if(SLpips > 0 && unrealizedPips <= -SLpips)
      {
         Print("SL損切り: ", Symbol(), " ", DoubleToStr(unrealizedPips, 1),
               "pips @", DoubleToStr(currentPrice, (int)MarketInfo(Symbol(), MODE_DIGITS)));
         ClosePosition();
         return;
      }
   }

   //=== タイムアウト判定: 毎ティック ===
   if(pos != 0)
   {
      datetime entryTime = GetEntryTime();
      double holdHours = (double)(TimeCurrent() - entryTime) / 3600.0;

      if(holdHours >= TimeoutH)
      {
         Print("タイムアウト決済: ", Symbol(), " 保有",
               DoubleToStr(holdHours, 1), "時間");
         ClosePosition();
         return;
      }
   }

   //=== Z決済判定: 5分足確定時 ===
   static datetime lastBar5m = 0;
   datetime currentBar5m = iTime(Symbol(), PERIOD_M5, 0);
   bool newBar5m = (currentBar5m != lastBar5m);
   if(newBar5m) lastBar5m = currentBar5m;

   if(pos != 0 && newBar5m)
   {
      double zscore = CalcZscore();  // 15分足ベースのZ
      if(zscore != 0)
      {
         if(pos == 1)
         {
            if(zscore > -ExitZ || zscore < -StopZ)
            {
               Print("決済(LONG): Z=", DoubleToStr(zscore, 3),
                     (zscore > -ExitZ ? " 利確" : " 損切りZ"));
               ClosePosition();
               pos = 0;
            }
         }
         else if(pos == -1)
         {
            if(zscore < ExitZ || zscore > StopZ)
            {
               Print("決済(SHORT): Z=", DoubleToStr(zscore, 3),
                     (zscore < ExitZ ? " 利確" : " 損切りZ"));
               ClosePosition();
               pos = 0;
            }
         }
      }
   }

   //=== エントリー判定: 15分足確定時のみ ===
   static datetime lastBar15m = 0;
   datetime currentBar15m = iTime(Symbol(), PERIOD_M15, 0);

   if(currentBar15m == lastBar15m) return;
   lastBar15m = currentBar15m;

   if(!IsTradeTime()) return;

   if(pos == 0)
   {
      double zscore = CalcZscore();
      if(zscore == 0) return;

      if(zscore > EntryZ)
      {
         Print("シグナル: SHORT Z=", DoubleToStr(zscore, 3));
         OpenPosition(-1);
      }
      else if(zscore < -EntryZ)
      {
         Print("シグナル: LONG Z=", DoubleToStr(zscore, 3));
         OpenPosition(1);
      }
   }
}
//+------------------------------------------------------------------+
