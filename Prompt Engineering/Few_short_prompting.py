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


class TechnicalAnalysisAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return """
You are a Technical Analysis Expert. Analyze stock data using:

REQUIRED ANALYSIS:
- Current price action and trends
- Key support/resistance levels
- RSI, MACD, Moving Averages
- Volume analysis
- Chart patterns

RESPONSE FORMAT:
1. Technical Status: [BULLISH/BEARISH/NEUTRAL]
2. Key Levels: Support: $X, Resistance: $Y
3. Indicators Summary: [Brief analysis]
4. Signal Strength: [1-10]
5. Risk Factors: [What could invalidate analysis]

Be specific with price levels and timeframes.
"""

    def process_query(self, question: str, stock_data: Dict[str, Any]) -> AgentResponse:
        prompt = f"""
{self.get_system_prompt()}

STOCK DATA: {stock_data}
USER QUESTION: {question}

Provide technical analysis with specific recommendations.
"""

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        return AgentResponse(
            agent_type="technical",
            response=response['message']['content'],
            confidence=0.8,
            data_used={"technical_indicators": stock_data.get("technical", {})}
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
                 analysis_model: str = "llama3.1:70b"):
        self.classifier = QuestionClassifier(classification_model)
        self.synthesizer = ResponseSynthesizer(analysis_model)

        # Initialize agents
        self.agents = {
            "technical": TechnicalAnalysisAgent(analysis_model),
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