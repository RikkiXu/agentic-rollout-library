#!/usr/bin/env python3
"""
Calculator tool for mathematical computations.
Based on the advanced calculator from examples but adapted for the core tool framework.
"""

import math
import ast
import operator
import logging
from typing import Any, Dict

from ..core.base_tool import AgenticBaseTool
from ..core.tool_schemas import OpenAIFunctionToolSchema, ToolResult, create_openai_tool_schema

logger = logging.getLogger(__name__)


class SafeMathEvaluator:
    """Safe mathematical expression evaluator."""
    
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    FUNCTIONS = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'ceil': math.ceil,
        'floor': math.floor,
        'factorial': math.factorial,
        'pi': math.pi,
        'e': math.e,
    }
    
    def evaluate(self, expression: str) -> float:
        """Safely evaluate mathematical expression."""
        try:
            expression = self._preprocess_expression(expression)
            node = ast.parse(expression, mode='eval')
            result = self._eval_node(node.body)
            return float(result)
        except Exception as e:
            raise ValueError(f"Expression evaluation failed: {str(e)}")
    
    def _preprocess_expression(self, expression: str) -> str:
        """Preprocess expression for evaluation."""
        expression = expression.replace(' ', '')
        replacements = {
            '×': '*', '÷': '/', '²': '**2', '³': '**3',
            'π': 'pi', 'Π': 'pi',
        }
        for old, new in replacements.items():
            expression = expression.replace(old, new)
        return expression
    
    def _eval_node(self, node):
        """Recursively evaluate AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in self.FUNCTIONS:
                return self.FUNCTIONS[node.id]
            else:
                raise ValueError(f"Unknown variable: {node.id}")
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.OPERATORS.get(type(node.op))
            if op:
                return op(left, right)
            else:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.OPERATORS.get(type(node.op))
            if op:
                return op(operand)
            else:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
        elif isinstance(node, ast.Call):
            func = self._eval_node(node.func)
            args = [self._eval_node(arg) for arg in node.args]
            if callable(func):
                return func(*args)
            else:
                raise ValueError(f"Not callable: {func}")
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")


class CalculatorTool(AgenticBaseTool):
    """Advanced calculator tool for mathematical computations."""
    
    def __init__(self, config: Dict = None):
        """Initialize calculator tool."""
        super().__init__(config)
        self.evaluator = SafeMathEvaluator()
        self.calculation_history = {}
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return OpenAI tool schema for calculator."""
        return create_openai_tool_schema(
            name="calculator",
            description="Advanced calculator for mathematical computations. Supports expressions, basic operations, scientific functions, and statistics.",
            parameters={
                "expression": {
                    "type": "string", 
                    "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sqrt(16) + sin(pi/2)')"
                },
                "operation": {
                    "type": "string",
                    "description": "Specific operation type",
                    "enum": ["add", "subtract", "multiply", "divide", "power", "sqrt", "factorial", "sin", "cos", "tan", "log"]
                },
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Array of numbers for list operations"
                },
                "a": {"type": "number", "description": "First operand for binary operations"},
                "b": {"type": "number", "description": "Second operand for binary operations"},
                "n": {"type": "number", "description": "Operand for unary operations"}
            },
            required=[]  # At least one parameter must be provided
        )
    
    async def _initialize_instance(self, instance_id: str, **kwargs) -> None:
        """Initialize instance-specific calculation history."""
        self.calculation_history[instance_id] = []
    
    async def execute_tool(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> ToolResult:
        """Execute calculator operations."""
        try:
            result = None
            calculation_type = None
            details = {}
            
            # Expression evaluation (highest priority)
            if 'expression' in parameters:
                expression = str(parameters['expression'])
                result = self.evaluator.evaluate(expression)
                calculation_type = "expression_evaluation"
                details = {"expression": expression}
            
            # Specific operations
            elif 'operation' in parameters:
                operation = parameters['operation'].lower()
                result, calculation_type, details = await self._perform_operation(operation, parameters)
            
            # List operations
            elif 'numbers' in parameters and isinstance(parameters['numbers'], list):
                numbers = [float(x) for x in parameters['numbers']]
                operation = parameters.get('operation', 'sum')
                result, calculation_type, details = await self._perform_list_operation(operation, numbers)
            
            # Binary operations
            elif 'a' in parameters and 'b' in parameters:
                a, b = float(parameters['a']), float(parameters['b'])
                operation = parameters.get('operation', 'add')
                result, calculation_type, details = await self._perform_binary_operation(operation, a, b)
            
            # Unary operations
            elif 'n' in parameters:
                n = float(parameters['n'])
                operation = parameters.get('operation', 'sqrt')
                result, calculation_type, details = await self._perform_unary_operation(operation, n)
            
            else:
                return ToolResult(success=False, error="No valid calculation parameters provided")
            
            # Format result
            if isinstance(result, float):
                if result.is_integer():
                    result = int(result)
                else:
                    result = round(result, 10)
            
            # Record calculation
            calculation_record = {
                "type": calculation_type,
                "input": parameters,
                "result": result,
                "details": details
            }
            
            if instance_id in self.calculation_history:
                self.calculation_history[instance_id].append(calculation_record)
            
            formatted_result = self._format_result(result, details)
            
            return ToolResult(
                success=True,
                result={
                    "result": result,
                    "formatted_result": formatted_result,
                    "calculation_type": calculation_type,
                    "details": details
                },
                metrics={"calculation_type": calculation_type}
            )
            
        except Exception as e:
            logger.error(f"Calculator execution failed: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _perform_operation(self, operation: str, params: Dict) -> tuple:
        """Perform specific mathematical operations."""
        if operation == "factorial":
            n = int(params.get('n', params.get('number', 1)))
            if n < 0:
                raise ValueError("Factorial cannot be computed for negative numbers")
            if n > 170:
                raise ValueError("Number too large for factorial computation")
            result = math.factorial(n)
            return result, "factorial", {"input": n}
        
        elif operation == "sqrt":
            n = float(params.get('n', params.get('number', 0)))
            if n < 0:
                raise ValueError("Cannot compute square root of negative number")
            result = math.sqrt(n)
            return result, "square_root", {"input": n}
        
        elif operation == "power":
            base = float(params.get('base', params.get('a', 0)))
            exponent = float(params.get('exponent', params.get('b', params.get('exp', 2))))
            result = base ** exponent
            return result, "power", {"base": base, "exponent": exponent}
        
        elif operation == "log":
            n = float(params.get('n', params.get('number', 1)))
            base = params.get('base', math.e)
            if n <= 0:
                raise ValueError("Logarithm input must be positive")
            if base == math.e:
                result = math.log(n)
            else:
                result = math.log(n, base)
            return result, "logarithm", {"input": n, "base": base}
        
        elif operation in ["sin", "cos", "tan"]:
            angle = float(params.get('angle', params.get('n', 0)))
            unit = params.get('unit', 'radians')
            if unit == 'degrees':
                angle = math.radians(angle)
            
            if operation == "sin":
                result = math.sin(angle)
            elif operation == "cos":
                result = math.cos(angle)
            elif operation == "tan":
                result = math.tan(angle)
            
            return result, f"trigonometric_{operation}", {"angle": angle, "unit": unit}
        
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    async def _perform_list_operation(self, operation: str, numbers: list) -> tuple:
        """Perform operations on number lists."""
        if operation == "sum":
            result = sum(numbers)
        elif operation in ["average", "mean"]:
            result = sum(numbers) / len(numbers)
        elif operation == "min":
            result = min(numbers)
        elif operation == "max":
            result = max(numbers)
        elif operation == "product":
            result = 1
            for num in numbers:
                result *= num
        elif operation in ["std", "standard_deviation"]:
            mean = sum(numbers) / len(numbers)
            variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
            result = math.sqrt(variance)
        else:
            raise ValueError(f"Unsupported list operation: {operation}")
        
        return result, f"list_{operation}", {"numbers": numbers, "count": len(numbers)}
    
    async def _perform_binary_operation(self, operation: str, a: float, b: float) -> tuple:
        """Perform binary operations."""
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Division by zero")
            result = a / b
        elif operation == "modulo":
            if b == 0:
                raise ValueError("Modulo by zero")
            result = a % b
        elif operation == "power":
            result = a ** b
        else:
            raise ValueError(f"Unsupported binary operation: {operation}")
        
        return result, f"binary_{operation}", {"a": a, "b": b}
    
    async def _perform_unary_operation(self, operation: str, n: float) -> tuple:
        """Perform unary operations."""
        if operation == "sqrt":
            if n < 0:
                raise ValueError("Cannot compute square root of negative number")
            result = math.sqrt(n)
        elif operation == "abs":
            result = abs(n)
        elif operation == "ceil":
            result = math.ceil(n)
        elif operation == "floor":
            result = math.floor(n)
        elif operation == "round":
            result = round(n, 0)
        elif operation == "factorial":
            if n < 0 or not n.is_integer():
                raise ValueError("Factorial only computes non-negative integers")
            result = math.factorial(int(n))
        else:
            raise ValueError(f"Unsupported unary operation: {operation}")
        
        return result, f"unary_{operation}", {"input": n}
    
    def _format_result(self, result, details: Dict) -> str:
        """Format result for display."""
        if isinstance(result, int):
            return str(result)
        elif isinstance(result, float):
            if abs(result) < 1e-10:
                return "0"
            elif abs(result) > 1e10:
                return f"{result:.2e}"
            else:
                return f"{result:.6f}".rstrip('0').rstrip('.')
        else:
            return str(result)
    
    async def _cleanup_instance(self, instance_id: str, **kwargs) -> None:
        """Clean up instance-specific data."""
        if instance_id in self.calculation_history:
            del self.calculation_history[instance_id]