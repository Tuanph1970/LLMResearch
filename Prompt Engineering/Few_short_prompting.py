from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re
import ollama


class QuestionCategory(Enum):
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    MARKET_TIMING = "market_timing"
    COMPANY_RESEARCH = "company_research"
    PORTFOLIO_MANAGEMENT = "portfolio_management"
    EDUCATIONAL = "educational"
    BUY_SELL_DECISION = "buy_sell_decision"


@dataclass
class RoutingResult:
    category: QuestionCategory
    confidence: float
    stock_symbol: Optional[str]
    required_agents: List[str]
    priority: int = 1


@dataclass
class AgentResponse:
    agent_type: str
    response: str
    confidence: float
    data_used: Dict[str, Any]


class BaseAgent(ABC):
    def __init__(self, model_name: str = "llama3.1:70b"):
        self.model_name = model_name

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def process_query(self, question: str, stock_data: Dict[str, Any]) -> AgentResponse:
        pass


class QuestionClassifier:
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = model_name
        self.classification_prompt = """
You are a Question Router for stock investment analysis. Classify the user question into ONE primary category.

CATEGORIES:
- TECHNICAL_ANALYSIS: Chart patterns, indicators, price movements, support/resistance
- FUNDAMENTAL_ANALYSIS: Financial metrics, company health, valuations, earnings
- RISK_ASSESSMENT: Portfolio risk, diversification, position sizing, volatility
- MARKET_TIMING: Entry/exit points, market conditions, when to buy/sell
- COMPANY_RESEARCH: Competitor analysis, growth prospects, news impact
- PORTFOLIO_MANAGEMENT: Asset allocation, hold/sell decisions, rebalancing
- BUY_SELL_DECISION: Direct buy/sell/hold recommendations
- EDUCATIONAL: How-to questions, concept explanations

STOCK_SYMBOLS: Extract any stock symbols mentioned (e.g., AAPL, TSLA, SPY)

Question: {question}

Respond in this exact format:
CATEGORY: [category_name]
CONFIDENCE: [0.0-1.0]
STOCK_SYMBOL: [symbol or NONE]
REASONING: [brief explanation]
"""

    def classify(self, question: str) -> RoutingResult:
        prompt = self.classification_prompt.format(question=question)

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        return self._parse_classification(response['message']['content'], question)

    def _parse_classification(self, response: str, original_question: str) -> RoutingResult:
        # Extract classification details
        category_match = re.search(r'CATEGORY:\s*(\w+)', response)
        confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response)
        symbol_match = re.search(r'STOCK_SYMBOL:\s*(\w+)', response)

        category_str = category_match.group(1) if category_match else "EDUCATIONAL"
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        stock_symbol = symbol_match.group(1) if symbol_match and symbol_match.group(1) != "NONE" else None

        # Map to enum
        try:
            category = QuestionCategory(category_str.lower())
        except ValueError:
            category = QuestionCategory.EDUCATIONAL

        # Determine required agents
        required_agents = self._get_required_agents(category, original_question)

        return RoutingResult(
            category=category,
            confidence=confidence,
            stock_symbol=stock_symbol,
            required_agents=required_agents
        )

    def _get_required_agents(self, category: QuestionCategory, question: str) -> List[str]:
        agent_mapping = {
            QuestionCategory.TECHNICAL_ANALYSIS: ["technical"],
            QuestionCategory.FUNDAMENTAL_ANALYSIS: ["fundamental"],
            QuestionCategory.RISK_ASSESSMENT: ["risk"],
            QuestionCategory.MARKET_TIMING: ["technical", "sentiment"],
            QuestionCategory.COMPANY_RESEARCH: ["fundamental", "news"],
            QuestionCategory.PORTFOLIO_MANAGEMENT: ["risk", "fundamental"],
            QuestionCategory.BUY_SELL_DECISION: ["technical", "fundamental", "risk"],
            QuestionCategory.EDUCATIONAL: ["educational"]
        }

        base_agents = agent_mapping.get(category, ["educational"])

        # Add conditional agents based on question content
        question_lower = question.lower()
        if any(word in question_lower for word in ["risk", "volatile", "safe"]):
            if "risk" not in base_agents:
                base_agents.append("risk")

        return base_agents


class AlgorithmicTechnicalAgent(BaseAgent):
    def __init__(self, model_name: str = None):
        # No LLM model needed for algorithmic analysis
        self.model_name = model_name

    def get_system_prompt(self) -> str:
        return "Algorithmic Technical Analysis Agent"

    def calculate_signal_strength(self, indicators: Dict[str, Any]) -> int:
        """Calculate overall signal strength (1-10)"""
        score = 5  # Neutral base

        # RSI contribution
        rsi = indicators.get("rsi", 50)
        if rsi > 70:
            score -= 2  # Overbought
        elif rsi < 30:
            score += 2  # Oversold
        elif 40 <= rsi <= 60:
            score += 1  # Healthy range

        # MACD contribution
        macd_data = indicators.get("macd", {})
        if macd_data.get("histogram", 0) > 0:
            score += 1
        else:
            score -= 1

        # Trend contribution
        trend = indicators.get("trend", "NEUTRAL")
        if trend == "BULLISH":
            score += 2
        elif trend == "BEARISH":
            score -= 2

        return max(1, min(10, score))
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0

        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Calculate MACD indicator"""
        if len(prices) < slow:
            return {"macd": 0, "signal": 0, "histogram": 0}

        # Simple EMA calculation
        def ema(data, period):
            multiplier = 2 / (period + 1)
            ema_values = [data[0]]
            for price in data[1:]:
                ema_values.append((price * multiplier) + (ema_values[-1] * (1 - multiplier)))
            return ema_values[-1]

        ema_fast = ema(prices, fast)
        ema_slow = ema(prices, slow)
        macd_line = ema_fast - ema_slow

        # Signal line would need more historical MACD values
        signal_line = macd_line * 0.9  # Simplified
        histogram = macd_line - signal_line

        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    def calculate_support_resistance(self, prices: List[float], volume: List[float] = None) -> Dict[str, float]:
        """Calculate support and resistance levels"""
        if len(prices) < 20:
            current_price = prices[-1] if prices else 100
            return {"support": current_price * 0.95, "resistance": current_price * 1.05}

        # Find local minima and maxima
        highs = []
        lows = []

        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                highs.append(prices[i])
            elif prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                lows.append(prices[i])

        resistance = max(highs[-3:]) if len(highs) >= 3 else max(prices) * 1.02
        support = min(lows[-3:]) if len(lows) >= 3 else min(prices) * 0.98

        return {"support": support, "resistance": resistance}

    def analyze_trend(self, prices: List[float]) -> str:
        """Determine trend direction"""
        if len(prices) < 10:
            return "NEUTRAL"

        # Simple moving averages
        sma_short = sum(prices[-5:]) / 5
        sma_long = sum(prices[-10:]) / 10

        if sma_short > sma_long * 1.02:
            return "BULLISH"
        elif sma_short < sma_long * 0.98:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def bollinger_bands(self, prices: List[float], period: int = 20, std_dev: int = 2) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            current = prices[-1] if prices else 100
            return {"upper": current * 1.1, "middle": current, "lower": current * 0.9}

        sma = sum(prices[-period:]) / period
        variance = sum((p - sma) ** 2 for p in prices[-period:]) / period
        std = variance ** 0.5

        return {
            "upper": sma + (std * std_dev),
            "middle": sma,
            "lower": sma - (std * std_dev)
        }

    def fibonacci_retracement(self, high: float, low: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        diff = high - low
        return {
            "level_236": high - (diff * 0.236),
            "level_382": high - (diff * 0.382),
            "level_500": high - (diff * 0.500),
            "level_618": high - (diff * 0.618),
            "level_786": high - (diff * 0.786)
        }

    def pattern_detection(self, prices: List[float]) -> List[str]:
        """Detect chart patterns"""
        patterns = []
        if len(prices) < 10:
            return patterns

        # Simple pattern detection
        recent = prices[-10:]

        # Double bottom
        if len(recent) >= 5:
            min_idx = recent.index(min(recent))
            if min_idx not in [0, len(recent) - 1]:
                patterns.append("POTENTIAL_DOUBLE_BOTTOM")

        # Ascending triangle
        highs = [recent[i] for i in range(1, len(recent) - 1)
                 if recent[i] > recent[i - 1] and recent[i] > recent[i + 1]]
        if len(highs) >= 2 and max(highs) - min(highs) < max(highs) * 0.02:
            patterns.append("ASCENDING_TRIANGLE")

        return patterns

    def volume_analysis(self, prices: List[float], volume: List[float]) -> Dict[str, Any]:
        """Analyze volume patterns"""
        if not volume or len(volume) < 10:
            return {"trend": "UNKNOWN", "strength": 5}

        avg_volume = sum(volume[-10:]) / 10
        recent_volume = volume[-1]
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

        return {
            "trend": "HIGH" if volume_ratio > 1.5 else "LOW" if volume_ratio < 0.5 else "NORMAL",
            "ratio": volume_ratio,
            "strength": min(10, max(1, int(volume_ratio * 5)))
        }
        """Calculate overall signal strength (1-10)"""
        score = 5  # Neutral base

        # RSI contribution
        rsi = indicators.get("rsi", 50)
        if rsi > 70:
            score -= 2  # Overbought
        elif rsi < 30:
            score += 2  # Oversold
        elif 40 <= rsi <= 60:
            score += 1  # Healthy range

        # MACD contribution
        macd_data = indicators.get("macd", {})
        if macd_data.get("histogram", 0) > 0:
            score += 1
        else:
            score -= 1

        # Trend contribution
        trend = indicators.get("trend", "NEUTRAL")
        if trend == "BULLISH":
            score += 2
        elif trend == "BEARISH":
            score -= 2

        return max(1, min(10, score))

    def process_query(self, question: str, stock_data: Dict[str, Any]) -> AgentResponse:
        # Extract price and volume data
        prices = stock_data.get("prices", [])
        volume = stock_data.get("volume", [])
        current_price = stock_data.get("current_price", prices[-1] if prices else 0)

        # Calculate technical indicators
        rsi = self.calculate_rsi(prices)
        macd = self.calculate_macd(prices)
        levels = self.calculate_support_resistance(prices, volume)
        trend = self.analyze_trend(prices)

        indicators = {
            "rsi": rsi,
            "macd": macd,
            "support": levels["support"],
            "resistance": levels["resistance"],
            "trend": trend
        }

        signal_strength = self.calculate_signal_strength(indicators)

        # Generate response
        response = f"""
TECHNICAL ANALYSIS RESULTS:

1. Technical Status: {trend}
2. Key Levels: Support: ${levels['support']:.2f}, Resistance: ${levels['resistance']:.2f}
3. Indicators Summary:
   - RSI: {rsi:.1f} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Normal'})
   - MACD: {macd['macd']:.3f} (Signal: {macd['signal']:.3f})
   - Current Price: ${current_price:.2f}
4. Signal Strength: {signal_strength}/10
5. Risk Factors: {'Price near resistance' if current_price > levels['resistance'] * 0.98 else 'Price near support' if current_price < levels['support'] * 1.02 else 'Price in normal range'}

RECOMMENDATION: {'BUY' if signal_strength >= 7 else 'SELL' if signal_strength <= 3 else 'HOLD'}
"""

        confidence = min(0.95, 0.6 + (abs(signal_strength - 5) * 0.07))

        return AgentResponse(
            agent_type="technical",
            response=response.strip(),
            confidence=confidence,
            data_used=indicators
        )


class FundamentalAnalysisAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return """
You are a Fundamental Analysis Expert. Analyze using:

FINANCIAL METRICS:
- P/E, P/B, PEG ratios vs industry
- Revenue/earnings growth trends
- Debt-to-equity, current ratio
- ROE, ROI, profit margins
- Dividend yield and sustainability

RESPONSE FORMAT:
1. Valuation: [UNDERVALUED/FAIRLY VALUED/OVERVALUED]
2. Financial Health: [1-10]
3. Growth Outlook: [STRONG/MODERATE/WEAK]
4. Key Strengths: [Top 3]
5. Key Concerns: [Top 3]
6. Fair Value: $X - $Y range

Focus on quantitative analysis with specific metrics.
"""

    def process_query(self, question: str, stock_data: Dict[str, Any]) -> AgentResponse:
        prompt = f"""
{self.get_system_prompt()}

FINANCIAL DATA: {stock_data}
USER QUESTION: {question}

Provide fundamental analysis with valuation assessment.
"""

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        return AgentResponse(
            agent_type="fundamental",
            response=response['message']['content'],
            confidence=0.85,
            data_used={"fundamentals": stock_data.get("fundamentals", {})}
        )


class RiskAssessmentAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return """
You are a Risk Management Specialist. Assess:

RISK FACTORS:
- Stock volatility (Beta, standard deviation)
- Portfolio concentration risk
- Sector/geographic diversification
- Correlation with market indices
- Maximum drawdown potential

RESPONSE FORMAT:
1. Risk Level: [LOW/MODERATE/HIGH]
2. Beta Analysis: [vs market]
3. Portfolio Impact: [concentration effect]
4. Position Size: [recommended % of portfolio]
5. Risk Mitigation: [stop-loss, hedging strategies]

Be conservative and emphasize risk management.
"""

    def process_query(self, question: str, stock_data: Dict[str, Any]) -> AgentResponse:
        prompt = f"""
{self.get_system_prompt()}

STOCK DATA: {stock_data}
USER QUESTION: {question}

Provide risk assessment with specific recommendations.
"""

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        return AgentResponse(
            agent_type="risk",
            response=response['message']['content'],
            confidence=0.9,
            data_used={"risk_metrics": stock_data.get("risk", {})}
        )


class ResponseSynthesizer:
    def __init__(self, model_name: str = "llama3.1:70b"):
        self.model_name = model_name

    def synthesize(self, agent_responses: List[AgentResponse], original_question: str,
                   user_context: Dict[str, Any] = None) -> str:
        synthesis_prompt = f"""
You are a Response Synthesizer. Combine multiple agent analyses into ONE coherent response.

AGENT OUTPUTS:
{self._format_agent_responses(agent_responses)}

USER QUESTION: {original_question}
USER CONTEXT: {user_context or {} }

SYNTHESIZE INTO:
1. Direct Answer: [Clear response to question]
2. Supporting Evidence: [Key points from each analysis]
3. Confidence Level: [HIGH/MEDIUM/LOW]
4. Action Items: [Specific next steps]
5. Disclaimers: [Risk warnings, not financial advice]

TONE: Professional but accessible, educational focus.
IMPORTANT: Always include "This is not financial advice" disclaimer.
"""

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": synthesis_prompt}]
        )

        return response['message']['content']

    def _format_agent_responses(self, responses: List[AgentResponse]) -> str:
        formatted = []
        for resp in responses:
            formatted.append(f"{resp.agent_type.upper()} ANALYSIS (Confidence: {resp.confidence}):\n{resp.response}\n")
        return "\n".join(formatted)


class StockAnalysisOrchestrator:
    def __init__(self, classification_model: str = "llama3.1:8b",
                 analysis_model: str = "llama3.1:8b"):
        self.classifier = QuestionClassifier(classification_model)
        self.synthesizer = ResponseSynthesizer(analysis_model)

        # Initialize agents
        self.agents = {
            "technical": AlgorithmicTechnicalAgent(),  # Uses custom algorithms instead of LLM
            "fundamental": FundamentalAnalysisAgent(analysis_model),
            "risk": RiskAssessmentAgent(analysis_model),
        }

    def process_question(self, question: str, stock_data: Dict[str, Any] = None,
                         user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        # Step 1: Classify the question
        routing_result = self.classifier.classify(question)

        # Step 2: Get stock data if symbol identified
        if routing_result.stock_symbol and not stock_data:
            stock_data = self._fetch_stock_data(routing_result.stock_symbol)

        # Step 3: Execute required agents
        agent_responses = []
        for agent_type in routing_result.required_agents:
            if agent_type in self.agents:
                response = self.agents[agent_type].process_query(question, stock_data or {})
                agent_responses.append(response)

        # Step 4: Synthesize final response
        final_response = self.synthesizer.synthesize(agent_responses, question, user_context)

        return {
            "response": final_response,
            "classification": routing_result,
            "agent_responses": agent_responses,
            "stock_symbol": routing_result.stock_symbol
        }

    def _fetch_stock_data(self, symbol: str) -> Dict[str, Any]:
        # Placeholder for actual stock data fetching
        # Integrate with your preferred data source (yfinance, Alpha Vantage, etc.)
        return {
            "symbol": symbol,
            "price": 150.0,
            "technical": {"rsi": 65, "macd": 0.5},
            "fundamentals": {"pe_ratio": 25, "debt_ratio": 0.3},
            "risk": {"beta": 1.2, "volatility": 0.25}
        }

    def add_agent(self, agent_type: str, agent: BaseAgent):
        """Add custom agents"""
        self.agents[agent_type] = agent


# Usage Example
if __name__ == "__main__":
    orchestrator = StockAnalysisOrchestrator()

    result = orchestrator.process_question(
        "Should I buy AAPL stock now? What's the technical analysis saying?",
        user_context={"risk_tolerance": "moderate", "timeframe": "6 months"}
    )

    print("Final Response:", result["response"])
    print("Classification:", result["classification"].category)
    print("Agents Used:", [r.agent_type for r in result["agent_responses"]])