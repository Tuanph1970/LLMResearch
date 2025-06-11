from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from openai import OpenAI
import json


class AIClientConfig:
    """Configuration class for AI client setup"""

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    @classmethod
    def for_openai(cls, api_key: str, model: str = "gpt-3.5-turbo"):
        """Create config for OpenAI"""
        return cls(api_key=api_key, model=model)

    @classmethod
    def for_ollama(cls, model: str = "llama2"):
        """Create config for local Ollama"""
        return cls(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            model=model
        )


class BaseFewShotPrompt(ABC):
    """Base class for few-shot prompting implementations"""

    def __init__(self, config: AIClientConfig):
        self.config = config
        self.client = self._setup_client()
        self.examples = []
        self.system_message = ""

    def _setup_client(self) -> OpenAI:
        """Setup OpenAI client based on configuration"""
        if self.config.base_url:
            return OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url
            )
        return OpenAI(api_key=self.config.api_key)

    @abstractmethod
    def add_example(self, input_text: str, expected_output: str) -> None:
        """Add a training example"""
        pass

    @abstractmethod
    def predict(self, input_text: str) -> str:
        """Make a prediction based on few-shot examples"""
        pass

    def clear_examples(self) -> None:
        """Clear all examples"""
        self.examples.clear()

    def get_example_count(self) -> int:
        """Get number of examples"""
        return len(self.examples)


class ConversationalFewShot(BaseFewShotPrompt):
    """Few-shot prompting using conversational message format"""

    def __init__(self, config: AIClientConfig, system_message: str = ""):
        super().__init__(config)
        self.system_message = system_message

    def add_example(self, input_text: str, expected_output: str) -> None:
        """Add example as user/assistant message pair"""
        self.examples.extend([
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": expected_output}
        ])

    def predict(self, input_text: str, max_tokens: int = 150, temperature: float = 0.1) -> str:
        """Make prediction using conversational format"""
        messages = []

        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})

        messages.extend(self.examples)
        messages.append({"role": "user", "content": input_text})

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"


class InlineFewShot(BaseFewShotPrompt):
    """Few-shot prompting using inline examples in single prompt"""

    def __init__(self, config: AIClientConfig, task_description: str = ""):
        super().__init__(config)
        self.task_description = task_description

    def add_example(self, input_text: str, expected_output: str) -> None:
        """Add example as input/output pair"""
        self.examples.append({
            "input": input_text,
            "output": expected_output
        })

    def predict(self, input_text: str, max_tokens: int = 150, temperature: float = 0.1) -> str:
        """Make prediction using inline format"""
        prompt = self._build_prompt(input_text)

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def _build_prompt(self, input_text: str) -> str:
        """Build the complete prompt with examples"""
        prompt = f"{self.task_description}\n\n" if self.task_description else ""

        if self.examples:
            prompt += "Examples:\n\n"
            for i, example in enumerate(self.examples, 1):
                prompt += f"Input: {example['input']}\n"
                prompt += f"Output: {example['output']}\n\n"

        prompt += f"Input: {input_text}\nOutput:"
        return prompt


class SentimentAnalyzer(ConversationalFewShot):
    """Sentiment analysis using few-shot learning"""

    def __init__(self, config: AIClientConfig):
        super().__init__(
            config,
            "You are a sentiment analysis expert. Classify text as Positive, Negative, or Neutral."
        )
        self._load_default_examples()

    def _load_default_examples(self):
        """Load default sentiment examples"""
        examples = [
            ("I absolutely love this new restaurant! The food is amazing.", "Positive"),
            ("This movie was terrible. Waste of time and money.", "Negative"),
            ("The weather is okay today, nothing special.", "Neutral"),
            ("The service was decent but the food could be better.", "Neutral"),
        ]

        for input_text, output in examples:
            self.add_example(input_text, output)


class EntityExtractor(InlineFewShot):
    """Named Entity Recognition using few-shot learning"""

    def __init__(self, config: AIClientConfig):
        super().__init__(
            config,
            "Extract person names, companies, and locations from the text. Format as JSON."
        )
        self._load_default_examples()

    def _load_default_examples(self):
        """Load default entity extraction examples"""
        examples = [
            (
                "John Smith works at Google in Mountain View, California.",
                '{"persons": "John Smith", "companies": "Google", "locations": "Mountain View California"}'
            ),
            (
                "Microsoft CEO Satya Nadella visited the Seattle headquarters.",
                '{"persons": "Satya Nadella", "companies": "Microsoft", "locations": "Seattle"}'
            ),
            (
                "Apple's Tim Cook announced the new iPhone at the Cupertino event.",
                '{"persons": "Tim Cook", "companies": "Apple", "locations": "Cupertino"}'
            )
        ]

        for input_text, output in examples:
            self.add_example(input_text, output)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities and return as dictionary"""
        result = self.predict(text)
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON response", "raw_response": result}


class TextClassifier(InlineFewShot):
    """Text classification using few-shot learning"""

    def __init__(self, config: AIClientConfig, categories: List[str] = None):
        super().__init__(
            config,
            f"Classify text into one of these categories: {', '.join(categories) if categories else 'predefined categories'}."
        )
        self.categories = categories or []

    def add_category_example(self, text: str, category: str):
        """Add example for specific category"""
        if category not in self.categories:
            self.categories.append(category)
        self.add_example(text, category)

    def load_support_examples(self):
        """Load customer support classification examples"""
        self.categories = ["Account Issues", "Shipping Inquiry", "Product Defect", "Return Request", "Product Inquiry"]

        examples = [
            ("I can't log into my account, forgot my password", "Account Issues"),
            ("When will my order arrive? I placed it 3 days ago", "Shipping Inquiry"),
            ("The product stopped working after 2 weeks", "Product Defect"),
            ("I want to return this item, it doesn't fit", "Return Request"),
            ("Do you have this item in size large?", "Product Inquiry")
        ]

        for text, category in examples:
            self.add_example(text, category)


class CodeExplainer(ConversationalFewShot):
    """Code explanation using few-shot learning"""

    def __init__(self, config: AIClientConfig):
        super().__init__(
            config,
            "Explain code snippets in simple terms for beginners."
        )
        self._load_default_examples()

    def _load_default_examples(self):
        """Load default code explanation examples"""
        examples = [
            (
                "def add_numbers(a, b):\n    return a + b",
                "This function takes two numbers as input and returns their sum. You call it like add_numbers(5, 3) to get 8."
            ),
            (
                "for i in range(5):\n    print(i)",
                "This loop prints numbers 0 through 4. The range(5) creates a sequence from 0 to 4, and the loop prints each number."
            ),
            (
                "if x > 10:\n    print('big')\nelse:\n    print('small')",
                "This checks if the variable x is greater than 10. If yes, it prints 'big', otherwise it prints 'small'."
            )
        ]

        for code, explanation in examples:
            self.add_example(code, explanation)


class DataFormatter(InlineFewShot):
    """Data format conversion using few-shot learning"""

    def __init__(self, config: AIClientConfig):
        super().__init__(
            config,
            "Convert unstructured text to structured JSON format."
        )
        self._load_default_examples()

    def _load_default_examples(self):
        """Load default data formatting examples"""
        examples = [
            (
                "John Doe, age 30, lives in New York, works as Engineer",
                '{"name": "John Doe", "age": 30, "city": "New York", "job": "Engineer"}'
            ),
            (
                "Sarah Smith, 25 years old, from Boston, Teacher",
                '{"name": "Sarah Smith", "age": 25, "city": "Boston", "job": "Teacher"}'
            ),
            (
                "Mike Johnson, aged 35, residing in Chicago, Software Developer",
                '{"name": "Mike Johnson", "age": 35, "city": "Chicago", "job": "Software Developer"}'
            )
        ]

        for input_text, output in examples:
            self.add_example(input_text, output)

    def format_to_json(self, text: str) -> Dict[str, Any]:
        """Format text to JSON and return as dictionary"""
        result = self.predict(text)
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON response", "raw_response": result}


class FewShotManager:
    """Manager class to handle multiple few-shot models"""

    def __init__(self, config: AIClientConfig):
        self.config = config
        self.models = {}

    def add_sentiment_analyzer(self, name: str = "sentiment") -> SentimentAnalyzer:
        """Add sentiment analyzer"""
        analyzer = SentimentAnalyzer(self.config)
        self.models[name] = analyzer
        return analyzer

    def add_entity_extractor(self, name: str = "entities") -> EntityExtractor:
        """Add entity extractor"""
        extractor = EntityExtractor(self.config)
        self.models[name] = extractor
        return extractor

    def add_text_classifier(self, name: str = "classifier", categories: List[str] = None) -> TextClassifier:
        """Add text classifier"""
        classifier = TextClassifier(self.config, categories)
        self.models[name] = classifier
        return classifier

    def add_code_explainer(self, name: str = "code") -> CodeExplainer:
        """Add code explainer"""
        explainer = CodeExplainer(self.config)
        self.models[name] = explainer
        return explainer

    def add_data_formatter(self, name: str = "formatter") -> DataFormatter:
        """Add data formatter"""
        formatter = DataFormatter(self.config)
        self.models[name] = formatter
        return formatter

    def get_model(self, name: str):
        """Get model by name"""
        return self.models.get(name)

    def list_models(self) -> List[str]:
        """List all available models"""
        return list(self.models.keys())


from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
import json
import re


class AIClientConfig:
    """Configuration class for AI client setup"""

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    @classmethod
    def for_openai(cls, api_key: str, model: str = "gpt-4"):
        return cls(api_key=api_key, model=model)

    @classmethod
    def for_ollama(cls, model: str = "llama2"):
        return cls(api_key="ollama", base_url="http://localhost:11434/v1", model=model)


@dataclass
class ComplexExample:
    """Structure for complex multi-part examples"""
    context: str
    input: str
    reasoning: str
    output: str
    metadata: Dict[str, Any] = None


class AdvancedFewShotPrompt:
    """Advanced few-shot prompting with complex reasoning"""

    def __init__(self, config: AIClientConfig, task_description: str):
        self.config = config
        self.client = self._setup_client()
        self.task_description = task_description
        self.examples: List[ComplexExample] = []
        self.reasoning_enabled = True

    def _setup_client(self) -> OpenAI:
        if self.config.base_url:
            return OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        return OpenAI(api_key=self.config.api_key)

    def add_complex_example(self, context: str, input: str, reasoning: str, output: str, metadata: Dict = None):
        """Add a complex example with reasoning steps"""
        example = ComplexExample(context, input, reasoning, output, metadata or {})
        self.examples.append(example)

    def _build_advanced_prompt(self, context: str, input: str) -> str:
        """Build sophisticated prompt with reasoning"""
        prompt = f"{self.task_description}\n\n"

        if self.reasoning_enabled:
            prompt += "For each example, I'll show the context, input, reasoning process, and final output.\n\n"

        # Add examples
        for i, example in enumerate(self.examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Context: {example.context}\n"
            prompt += f"Input: {example.input}\n"

            if self.reasoning_enabled and example.reasoning:
                prompt += f"Reasoning: {example.reasoning}\n"

            prompt += f"Output: {example.output}\n\n"

        # Add current task
        prompt += f"Now solve this:\n"
        prompt += f"Context: {context}\n"
        prompt += f"Input: {input}\n"

        if self.reasoning_enabled:
            prompt += f"Reasoning: Let me think step by step...\n"

        prompt += f"Output:"

        return prompt

    def predict_with_reasoning(self, context: str, input: str, max_tokens: int = 500, temperature: float = 0.1) -> Dict[
        str, str]:
        """Generate prediction with reasoning steps"""
        prompt = self._build_advanced_prompt(context, input)

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )

            result = response.choices[0].message.content.strip()

            # Try to parse reasoning and output
            if "Reasoning:" in result:
                parts = result.split("Output:", 1)
                if len(parts) == 2:
                    reasoning = parts[0].replace("Reasoning:", "").strip()
                    output = parts[1].strip()
                    return {"reasoning": reasoning, "output": output, "raw": result}

            return {"reasoning": "", "output": result, "raw": result}

        except Exception as e:
            return {"error": str(e), "reasoning": "", "output": "", "raw": ""}


class SQLQueryGenerator(AdvancedFewShotPrompt):
    """Complex SQL query generation from natural language with business logic"""

    def __init__(self, config: AIClientConfig):
        super().__init__(
            config,
            "Generate SQL queries from natural language requests. Consider business logic, performance, and edge cases."
        )
        self._load_sql_examples()

    def _load_sql_examples(self):
        """Load complex SQL examples with business reasoning"""

        # Example 1: Complex JOIN with business logic
        self.add_complex_example(
            context="E-commerce database with tables: customers (id, name, email, registration_date), orders (id, customer_id, order_date, total_amount, status), products (id, name, price, category), order_items (order_id, product_id, quantity, unit_price)",
            input="Find customers who spent more than $1000 in the last 6 months but haven't ordered anything in the last 30 days",
            reasoning="""1. Need to identify time periods: last 6 months and last 30 days
2. First find customers with >$1000 spending in last 6 months
3. Then exclude those who ordered in last 30 days
4. Use LEFT JOIN to find customers without recent orders
5. Consider only completed orders (status = 'completed')
6. Group by customer to sum their spending""",
            output="""SELECT DISTINCT c.id, c.name, c.email, 
       SUM(o.total_amount) as total_spent
FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE o.order_date >= DATE_SUB(NOW(), INTERVAL 6 MONTH)
  AND o.status = 'completed'
  AND c.id NOT IN (
    SELECT DISTINCT customer_id 
    FROM orders 
    WHERE order_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
      AND status = 'completed'
  )
GROUP BY c.id, c.name, c.email
HAVING SUM(o.total_amount) > 1000
ORDER BY total_spent DESC;"""
        )

        # Example 2: Window functions with complex business logic
        self.add_complex_example(
            context="Sales database with tables: sales (id, salesperson_id, sale_date, amount, region), employees (id, name, department, hire_date)",
            input="Show each salesperson's performance compared to their regional average, including their ranking within the region",
            reasoning="""1. Need window functions to calculate regional averages and rankings
2. Calculate average sales per region for comparison
3. Rank salespeople within their region
4. Show both absolute performance and relative performance
5. Consider only active salespeople (those with recent sales)""",
            output="""SELECT 
    e.name as salesperson,
    s.region,
    SUM(s.amount) as total_sales,
    AVG(SUM(s.amount)) OVER (PARTITION BY s.region) as regional_avg,
    SUM(s.amount) - AVG(SUM(s.amount)) OVER (PARTITION BY s.region) as vs_regional_avg,
    RANK() OVER (PARTITION BY s.region ORDER BY SUM(s.amount) DESC) as regional_rank,
    COUNT(*) OVER (PARTITION BY s.region) as salespeople_in_region
FROM employees e
JOIN sales s ON e.id = s.salesperson_id
WHERE s.sale_date >= DATE_SUB(NOW(), INTERVAL 12 MONTH)
GROUP BY e.id, e.name, s.region
ORDER BY s.region, regional_rank;"""
        )

        # Example 3: Complex analytics query
        self.add_complex_example(
            context="Subscription service with tables: users (id, email, signup_date), subscriptions (id, user_id, plan_id, start_date, end_date, status), plans (id, name, price, billing_cycle)",
            input="Calculate monthly recurring revenue (MRR) trend and churn rate for the last 12 months",
            reasoning="""1. MRR calculation needs to account for different billing cycles
2. Need to normalize all subscriptions to monthly values
3. Churn rate = customers who cancelled / total active customers
4. Generate time series for last 12 months
5. Handle pro-ration for annual plans
6. Exclude paused/cancelled subscriptions from active count""",
            output="""WITH monthly_series AS (
  SELECT DATE_FORMAT(DATE_SUB(NOW(), INTERVAL n MONTH), '%Y-%m-01') as month_start
  FROM (SELECT 0 n UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION 
        SELECT 4 UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION 
        SELECT 8 UNION SELECT 9 UNION SELECT 10 UNION SELECT 11) months
),
monthly_mrr AS (
  SELECT 
    ms.month_start,
    SUM(CASE 
      WHEN p.billing_cycle = 'monthly' THEN p.price
      WHEN p.billing_cycle = 'annual' THEN p.price / 12
      ELSE 0 
    END) as mrr,
    COUNT(DISTINCT s.user_id) as active_subscribers
  FROM monthly_series ms
  LEFT JOIN subscriptions s ON s.start_date <= LAST_DAY(ms.month_start)
    AND (s.end_date IS NULL OR s.end_date > ms.month_start)
    AND s.status = 'active'
  LEFT JOIN plans p ON s.plan_id = p.id
  GROUP BY ms.month_start
),
churn_calc AS (
  SELECT 
    DATE_FORMAT(s.end_date, '%Y-%m-01') as churn_month,
    COUNT(DISTINCT s.user_id) as churned_users
  FROM subscriptions s
  WHERE s.end_date >= DATE_SUB(NOW(), INTERVAL 12 MONTH)
    AND s.status = 'cancelled'
  GROUP BY DATE_FORMAT(s.end_date, '%Y-%m-01')
)
SELECT 
  m.month_start,
  m.mrr,
  m.active_subscribers,
  COALESCE(c.churned_users, 0) as churned_users,
  CASE 
    WHEN m.active_subscribers > 0 
    THEN (COALESCE(c.churned_users, 0) * 100.0 / m.active_subscribers)
    ELSE 0 
  END as churn_rate_percent
FROM monthly_mrr m
LEFT JOIN churn_calc c ON m.month_start = c.churn_month
ORDER BY m.month_start;"""
        )


class FinancialAnalyzer(AdvancedFewShotPrompt):
    """Complex financial analysis and investment recommendations"""

    def __init__(self, config: AIClientConfig):
        super().__init__(
            config,
            "Analyze financial data and provide investment recommendations with detailed reasoning."
        )
        self._load_financial_examples()

    def _load_financial_examples(self):
        """Load complex financial analysis examples"""

        self.add_complex_example(
            context="Company Analysis: TechCorp Inc. - Software company, Market Cap: $50B, Current Stock Price: $125",
            input="Q3 Results: Revenue $2.1B (+15% YoY), Net Income $420M (+22% YoY), Free Cash Flow $380M (+18% YoY), Debt-to-Equity: 0.3, P/E Ratio: 28, Industry Average P/E: 25",
            reasoning="""1. Revenue Growth Analysis: 15% YoY is strong for mature tech company
2. Profitability Metrics: 22% net income growth > revenue growth indicates improving margins
3. Cash Generation: Strong FCF of $380M shows real cash-generating ability
4. Debt Management: D/E of 0.3 is conservative, indicates financial stability
5. Valuation: P/E of 28 vs industry 25 suggests slight premium, but justified by growth
6. Quality Indicators: FCF/Net Income ratio of 90% shows high earnings quality
7. Risk Assessment: Tech sector volatility, but fundamentals strong""",
            output="""{
  "recommendation": "BUY",
  "target_price": "$145",
  "confidence": "High",
  "key_strengths": [
    "Strong revenue growth of 15% YoY",
    "Expanding profit margins (22% net income growth)",
    "Excellent cash generation ($380M FCF)",
    "Conservative debt levels (D/E: 0.3)"
  ],
  "concerns": [
    "Trading at premium to industry P/E",
    "Tech sector volatility risk"
  ],
  "catalysts": [
    "Continued margin expansion",
    "Strong cash flow for reinvestment",
    "Potential for multiple expansion"
  ],
  "price_targets": {
    "bull_case": "$160",
    "base_case": "$145", 
    "bear_case": "$110"
  }
}"""
        )

        self.add_complex_example(
            context="Portfolio Allocation: Conservative Investor, Age 55, $500K portfolio, Risk Tolerance: Low-Medium, Goal: Retirement in 10 years",
            input="Current Allocation: 60% Stocks, 40% Bonds. Market Conditions: Rising interest rates, inflation at 4%, stock market volatility high",
            reasoning="""1. Time Horizon: 10 years allows for some equity exposure
2. Risk Profile: Conservative suggests bond-heavy allocation
3. Inflation Impact: 4% inflation erodes bond real returns
4. Rising Rates: Bad for existing bonds, good for new bond purchases
5. Age Factor: Rule of thumb suggests 45% stocks (100-55), current 60% is aggressive
6. Volatility Consideration: High volatility suggests defensive positioning
7. Diversification Needs: Consider TIPS, international exposure, alternatives""",
            output="""{
  "recommended_allocation": {
    "stocks": "45%",
    "bonds": "35%", 
    "alternatives": "15%",
    "cash": "5%"
  },
  "specific_recommendations": {
    "stocks": {
      "us_large_cap": "25%",
      "international_developed": "15%",
      "emerging_markets": "5%"
    },
    "bonds": {
      "treasury_tips": "15%",
      "corporate_bonds": "10%",
      "short_term_bonds": "10%"
    },
    "alternatives": {
      "reits": "8%",
      "commodities": "4%",
      "infrastructure": "3%"
    }
  },
  "rationale": "Reduced equity allocation for age-appropriate risk, added TIPS for inflation protection, included alternatives for diversification",
  "rebalancing_triggers": [
    "Quarterly review",
    "5% deviation from target allocation",
    "Major life events"
  ]
}"""
        )


class CodeReviewer(AdvancedFewShotPrompt):
    """Advanced code review with security, performance, and architecture analysis"""

    def __init__(self, config: AIClientConfig):
        super().__init__(
            config,
            "Perform comprehensive code reviews focusing on security, performance, maintainability, and best practices."
        )
        self._load_code_examples()

    def _load_code_examples(self):
        """Load complex code review examples"""

        self.add_complex_example(
            context="Python Flask web application - User authentication endpoint",
            input="""@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    user = User.query.filter_by(username=username).first()
    if user and user.password == password:
        session['user_id'] = user.id
        return redirect('/dashboard')
    else:
        return 'Invalid credentials', 401""",
            reasoning="""1. Security Issues:
   - Plain text password comparison (no hashing)
   - No CSRF protection
   - No rate limiting for brute force attacks
   - SQL injection potential if not using ORM properly
   - Session fixation vulnerability

2. Input Validation:
   - No validation of username/password format
   - No sanitization of inputs
   - Missing error handling for missing form fields

3. Performance Issues:
   - Database query on every login attempt
   - No caching mechanism

4. Architecture Concerns:
   - Business logic mixed with routing
   - No separation of concerns
   - Hard to test and maintain""",
            output="""{
  "severity": "CRITICAL",
  "issues": [
    {
      "type": "SECURITY",
      "level": "CRITICAL",
      "description": "Plain text password storage and comparison",
      "line": 6,
      "recommendation": "Use bcrypt or similar for password hashing"
    },
    {
      "type": "SECURITY", 
      "level": "HIGH",
      "description": "No CSRF protection",
      "line": 1,
      "recommendation": "Add CSRF tokens to forms"
    },
    {
      "type": "SECURITY",
      "level": "HIGH", 
      "description": "No rate limiting",
      "line": 1,
      "recommendation": "Implement rate limiting for login attempts"
    },
    {
      "type": "VALIDATION",
      "level": "MEDIUM",
      "description": "Missing input validation",
      "line": 3-4,
      "recommendation": "Validate username/password format and presence"
    }
  ],
  "improved_code": "```python\\nfrom flask_limiter import Limiter\\nfrom werkzeug.security import check_password_hash\\nfrom flask_wtf.csrf import validate_csrf\\n\\n@app.route('/login', methods=['POST'])\\n@limiter.limit('5 per minute')\\ndef login():\\n    try:\\n        validate_csrf(request.form.get('csrf_token'))\\n        \\n        username = request.form.get('username', '').strip()\\n        password = request.form.get('password', '')\\n        \\n        if not username or not password:\\n            return jsonify({'error': 'Username and password required'}), 400\\n            \\n        user = User.query.filter_by(username=username).first()\\n        \\n        if user and check_password_hash(user.password_hash, password):\\n            session.regenerate()  # Prevent session fixation\\n            session['user_id'] = user.id\\n            return redirect('/dashboard')\\n        else:\\n            # Log failed attempt\\n            logger.warning(f'Failed login attempt for {username}')\\n            return jsonify({'error': 'Invalid credentials'}), 401\\n            \\n    except ValidationError:\\n        return jsonify({'error': 'Invalid CSRF token'}), 403\\n    except Exception as e:\\n        logger.error(f'Login error: {e}')\\n        return jsonify({'error': 'Internal server error'}), 500\\n```",
  "additional_recommendations": [
    "Implement account lockout after multiple failed attempts",
    "Add two-factor authentication",
    "Use HTTPS only for authentication endpoints",
    "Implement proper session management",
    "Add comprehensive logging and monitoring"
  ]
}"""
        )


class ResearchAnalyzer(AdvancedFewShotPrompt):
    """Complex research paper analysis and synthesis"""

    def __init__(self, config: AIClientConfig):
        super().__init__(
            config,
            "Analyze research papers and synthesize findings with proper citations and critical evaluation."
        )
        self._load_research_examples()

    def _load_research_examples(self):
        """Load complex research analysis examples"""

        self.add_complex_example(
            context="Machine Learning Research - Topic: Large Language Model Training Efficiency",
            input="""Papers to analyze:
1. "Scaling Laws for Neural Language Models" - Kaplan et al. (2020) - Shows relationship between model size, dataset size, and performance
2. "Training Compute-Optimal Large Language Models" - Hoffmann et al. (2022) - Introduces Chinchilla scaling laws
3. "PaLM: Scaling Language Modeling with Pathways" - Chowdhery et al. (2022) - 540B parameter model with architecture innovations

Question: What are the key findings about optimal resource allocation for LLM training?""",
            reasoning="""1. Literature Review: Need to extract key findings from each paper
2. Conflicting Views: Kaplan vs Hoffmann scaling laws show evolution of understanding
3. Methodological Analysis: Compare experimental setups and datasets
4. Synthesis: Identify convergent themes and contradictions
5. Practical Implications: What do findings mean for practitioners
6. Limitations: Acknowledge constraints and gaps in research
7. Future Directions: Identify areas needing more research""",
            output="""{
  "synthesis": "Research evolution shows shifting understanding of optimal LLM training resource allocation",
  "key_findings": {
    "scaling_relationships": {
      "kaplan_2020": "Advocated for larger models with relatively smaller datasets (power law: model size scales faster than data)",
      "hoffmann_2022": "Challenged Kaplan's findings, showing compute-optimal training requires balanced scaling of model size and training tokens",
      "practical_impact": "Chinchilla scaling suggests many large models are undertrained rather than oversized"
    },
    "resource_allocation": {
      "kaplan_approach": "Allocate most compute to model parameters",
      "chinchilla_approach": "Balance compute between model size and training data volume",
      "efficiency_gains": "Chinchilla-optimal models achieve better performance per unit of compute"
    }
  },
  "convergent_themes": [
    "Compute budget optimization is critical for practical deployment",
    "Data quality and quantity both matter significantly", 
    "Architecture innovations can improve efficiency (PaLM's sparsity)",
    "Scaling laws provide guidelines but need empirical validation"
  ],
  "contradictions": [
    "Optimal scaling ratios differ between studies due to different evaluation criteria",
    "PaLM results suggest architecture changes can break standard scaling relationships"
  ],
  "limitations": [
    "Most studies focus on English language tasks",
    "Limited evaluation on downstream task performance vs perplexity",
    "Computational constraints limit exploration of largest scales"
  ],
  "practical_implications": {
    "for_researchers": "Invest equally in data curation and model architecture",
    "for_industry": "Smaller, well-trained models may outperform larger undertrained ones",
    "for_policy": "Compute efficiency has environmental and accessibility implications"
  },
  "future_research": [
    "Multi-modal scaling relationships",
    "Task-specific optimal scaling",
    "Environmental impact optimization",
    "Scaling laws for specialized domains"
  ]
}"""
        )


class BusinessProcessAnalyzer(AdvancedFewShotPrompt):
    """Complex business process optimization and automation analysis"""

    def __init__(self, config: AIClientConfig):
        super().__init__(
            config,
            "Analyze business processes and recommend optimization strategies with ROI calculations."
        )
        self._load_business_examples()

    def _load_business_examples(self):
        """Load complex business process examples"""

        self.add_complex_example(
            context="Customer Support Process - Mid-size SaaS company, 500 tickets/day, 12 support agents, Average resolution time: 8 hours",
            input="""Current Process:
1. Customer submits ticket via email/portal
2. Manual triage by Level 1 agent (30 min avg)
3. Assignment to appropriate team (15 min avg)
4. Investigation and resolution (6 hours avg)
5. Customer response and closure (1 hour avg)

Issues:
- 40% of tickets are repetitive (password resets, billing questions)
- Triage often misclassifies tickets (20% reassigned)
- Knowledge base exists but underutilized
- No SLA tracking
- Customer satisfaction: 3.2/5""",
            reasoning="""1. Process Mapping: Identify bottlenecks and inefficiencies
2. Volume Analysis: 40% repetitive = 200 tickets/day automation opportunity
3. Cost Analysis: 12 agents × 8 hours × $25/hour = $2400/day in labor costs
4. Error Analysis: 20% misclassification causes delays and rework
5. Technology Assessment: Existing knowledge base underutilized
6. ROI Calculation: Automation potential vs implementation costs
7. Change Management: Consider agent training and customer adaptation""",
            output="""{
  "optimization_strategy": {
    "immediate_wins": {
      "chatbot_implementation": {
        "target": "Handle 80% of repetitive tickets (160/day)",
        "estimated_savings": "$960/day in agent time",
        "implementation_cost": "$50,000",
        "payback_period": "52 days",
        "impact": "Reduces agent workload by 30%"
      },
      "improved_triage": {
        "solution": "AI-powered ticket classification",
        "target": "Reduce misclassification from 20% to 5%",
        "savings": "$300/day in rework costs",
        "implementation_cost": "$25,000",
        "payback_period": "83 days"
      }
    },
    "medium_term": {
      "knowledge_base_enhancement": {
        "solution": "Intelligent article suggestions during ticket creation",
        "target": "20% self-service resolution increase",
        "savings": "$480/day",
        "implementation_cost": "$30,000",
        "payback_period": "62 days"
      },
      "workflow_automation": {
        "solution": "Automated escalation and SLA tracking",
        "target": "Reduce average resolution time to 6 hours",
        "customer_satisfaction_impact": "Expected increase to 4.0/5"
      }
    }
  },
  "implementation_roadmap": {
    "phase_1": "Chatbot + basic automation (Month 1-2)",
    "phase_2": "AI triage + knowledge base (Month 3-4)", 
    "phase_3": "Advanced workflow automation (Month 5-6)"
  },
  "roi_analysis": {
    "total_investment": "$105,000",
    "annual_savings": "$438,000",
    "roi": "317%",
    "break_even": "3.5 months"
  },
  "success_metrics": [
    "Average resolution time: 8h → 6h",
    "Customer satisfaction: 3.2 → 4.0",
    "Agent productivity: +40%",
    "First contact resolution: +25%"
  ],
  "risks_and_mitigations": {
    "customer_adoption": "Gradual rollout with opt-out options",
    "agent_resistance": "Comprehensive training and repositioning as value-add",
    "technical_integration": "Phased implementation with rollback plans"
  }
}"""
        )


def demonstrate_complex_examples():
    """Demonstrate the complex few-shot prompting examples"""

    # Setup (choose your configuration)
    config = AIClientConfig.for_ollama("llama2")  # or AIClientConfig.for_openai("your-key")

    print("=== Complex Few-Shot Prompting Demonstrations ===\n")

    # SQL Query Generation
    print("1. SQL Query Generation:")
    print("-" * 40)
    sql_gen = SQLQueryGenerator(config)

    result = sql_gen.predict_with_reasoning(
        context="Inventory database with tables: products (id, name, category, price, stock_quantity), suppliers (id, name, country), product_suppliers (product_id, supplier_id, cost, lead_time_days)",
        input="Find products that are low in stock (less than 20 units) and identify the fastest supplier for each, considering both cost and lead time"
    )

    print(f"Generated SQL:\n{result.get('output', 'Error occurred')}")
    print(f"\nReasoning:\n{result.get('reasoning', 'No reasoning provided')}")

    print("\n" + "=" * 60 + "\n")

    # Financial Analysis
    print("2. Financial Analysis:")
    print("-" * 40)
    fin_analyzer = FinancialAnalyzer(config)

    result = fin_analyzer.predict_with_reasoning(
        context="Company Analysis: GreenTech Solutions - Clean energy startup, Market Cap: $2B, Current Stock Price: $45",
        input="Q2 Results: Revenue $150M (+85% YoY), Net Loss $20M (improved from $45M loss), Cash $200M, Burn Rate $30M/quarter, P/S Ratio: 5.3, Sector Average P/S: 3.8"
    )

    print(f"Financial Recommendation:\n{result.get('output', 'Error occurred')}")
    print(f"\nAnalysis Process:\n{result.get('reasoning', 'No reasoning provided')}")

    print("\n" + "=" * 60 + "\n")

    # Code Review
    print("3. Code Security Review:")
    print("-" * 40)
    code_reviewer = CodeReviewer(config)

    sample_code = """def transfer_funds(from_account, to_account, amount):
    from_balance = get_balance(from_account)
    if from_balance >= amount:
        update_balance(from_account, from_balance - amount)
        to_balance = get_balance(to_account)
        update_balance(to_account, to_balance + amount)
        return True
    return False"""

    result = code_reviewer.predict_with_reasoning(
        context="Banking application - Fund transfer function",
        input=sample_code
    )

    print(f"Code Review Results:\n{result.get('output', 'Error occurred')}")
    print(f"\nSecurity Analysis:\n{result.get('reasoning', 'No reasoning provided')}")





# Example usage and demonstration
def main():
    

    # Setup configuration (choose one)
    # config = AIClientConfig.for_openai("your-api-key-here")
    config = AIClientConfig.for_ollama("llama2")  # For local Ollama

    print("=== Few-Shot Prompting with Classes ===\n")

    # Method 1: Using individual classes
    print("1. Individual Classes Demo:")

    # Sentiment Analysis
    sentiment = SentimentAnalyzer(config)
    result = sentiment.predict("This is the best day of my life!")
    print(f"Sentiment: {result}")

    # Entity Extraction
    entities = EntityExtractor(config)
    result = entities.extract_entities("Amazon founder Jeff Bezos spoke in New York about Blue Origin.")
    print(f"Entities: {result}")

    # Text Classification
    classifier = TextClassifier(config)
    classifier.load_support_examples()
    result = classifier.predict("I received the wrong item in my package")
    print(f"Classification: {result}")

    print("\n" + "=" * 50 + "\n")

    # Method 2: Using FewShotManager
    print("2. Manager Class Demo:")

    manager = FewShotManager(config)

    # Add models
    sentiment = manager.add_sentiment_analyzer()
    entities = manager.add_entity_extractor()
    formatter = manager.add_data_formatter()

    print(f"Available models: {manager.list_models()}")

    # Use models
    result = manager.get_model("sentiment").predict("I love this new feature!")
    print(f"Sentiment: {result}")

    result = manager.get_model("formatter").format_to_json("Emma Wilson, 28 years old, from Seattle, Marketing Manager")
    print(f"Formatted data: {result}")

    print("\n" + "=" * 50 + "\n")

    # Method 3: Custom classifier
    print("3. Custom Classifier Demo:")

    custom_classifier = TextClassifier(config, ["Technology", "Sports", "Politics", "Entertainment"])

    # Add custom examples
    custom_classifier.add_category_example("New iPhone released with amazing features", "Technology")
    custom_classifier.add_category_example("Lakers won the championship game", "Sports")
    custom_classifier.add_category_example("President announces new policy", "Politics")
    custom_classifier.add_category_example("New movie breaks box office records", "Entertainment")

    result = custom_classifier.predict("Apple announces revolutionary AI chip")
    print(f"Custom classification: {result}")


if __name__ == "__main__":
    try:
        main()
        demonstrate_complex_examples()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Install openai: pip install openai")
        print("2. Set your API key or configure for Ollama")
        print("3. Have the model available (for Ollama: ollama pull llama2)")

