import networkx as nx
from typing import Dict, List, Set, Tuple, Any, Optional
import json
import os
from pydantic import BaseModel


class SymbolicRule(BaseModel):
    """
    Represents a logical rule in the symbolic reasoning system.
    """
    name: str
    conditions: List[Dict[str, str]]
    actions: List[Dict[str, str]]
    
    def applies_to(self, facts: Dict[str, Any]) -> bool:
        """
        Determine if this rule applies to the given facts.
        
        Args:
            facts: Dictionary of facts
            
        Returns:
            True if all conditions of this rule are satisfied, False otherwise
        """
        for condition in self.conditions:
            subject = condition.get('subject', '')
            predicate = condition.get('predicate', '')
            object = condition.get('object', '')
            
            # Check if this triplet exists in facts
            if subject in facts:
                if predicate == 'is_a':
                    # Handle 'is_a' relationship
                    if isinstance(facts[subject], dict) and 'type' in facts[subject]:
                        if facts[subject]['type'] != object:
                            return False
                    else:
                        return False
                elif predicate == 'has_property':
                    # Handle properties
                    if isinstance(facts[subject], dict) and 'properties' in facts[subject]:
                        if object not in facts[subject]['properties']:
                            return False
                    else:
                        return False
                else:
                    # Handle general predicates
                    if predicate not in facts[subject]:
                        return False
                    if facts[subject][predicate] != object:
                        return False
            else:
                return False
                
        return True
    
    def apply(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply this rule to infer new facts.
        
        Args:
            facts: Dictionary of existing facts
            
        Returns:
            Dictionary of facts with inferred facts added
        """
        if not self.applies_to(facts):
            return facts
        
        new_facts = facts.copy()
        
        for action in self.actions:
            subject = action.get('subject', '')
            predicate = action.get('predicate', '')
            object = action.get('object', '')
            
            if subject not in new_facts:
                new_facts[subject] = {}
                
            if predicate == 'is_a':
                # Set type
                if isinstance(new_facts[subject], dict):
                    new_facts[subject]['type'] = object
                else:
                    new_facts[subject] = {'type': object}
            elif predicate == 'has_property':
                # Add property
                if isinstance(new_facts[subject], dict):
                    if 'properties' not in new_facts[subject]:
                        new_facts[subject]['properties'] = []
                    if object not in new_facts[subject]['properties']:
                        new_facts[subject]['properties'].append(object)
                else:
                    new_facts[subject] = {'properties': [object]}
            else:
                # Set general predicate
                if isinstance(new_facts[subject], dict):
                    new_facts[subject][predicate] = object
                else:
                    new_facts[subject] = {predicate: object}
        
        return new_facts


class SymbolicReasoningEngine:
    """
    Symbolic Reasoning Engine that applies logical rules to facts.
    """
    
    def __init__(self, rules_file: Optional[str] = None):
        """
        Initialize the symbolic reasoning engine.
        
        Args:
            rules_file: Path to JSON file containing rules
        """
        self.rules: List[SymbolicRule] = []
        self.knowledge_graph = nx.DiGraph()
        
        # Load rules from file if provided
        if rules_file and os.path.exists(rules_file):
            self.load_rules(rules_file)
        else:
            # Add some default rules
            self._add_default_rules()
    
    def _add_default_rules(self):
        """Add default reasoning rules."""
        # Rule: If an object is a cat, then it's a pet
        self.add_rule(SymbolicRule(
            name="cat_is_pet",
            conditions=[{"subject": "detected_object", "predicate": "is_a", "object": "cat"}],
            actions=[{"subject": "detected_object", "predicate": "is_a", "object": "pet"},
                    {"subject": "image", "predicate": "contains", "object": "pet"}]
        ))
        
        # Rule: If an object is a dog, then it's a pet
        self.add_rule(SymbolicRule(
            name="dog_is_pet",
            conditions=[{"subject": "detected_object", "predicate": "is_a", "object": "dog"}],
            actions=[{"subject": "detected_object", "predicate": "is_a", "object": "pet"},
                    {"subject": "image", "predicate": "contains", "object": "pet"}]
        ))
        
        # Rule: If an object is both alive and owned by humans, it's a pet
        self.add_rule(SymbolicRule(
            name="living_owned_is_pet",
            conditions=[
                {"subject": "detected_object", "predicate": "has_property", "object": "alive"},
                {"subject": "detected_object", "predicate": "has_property", "object": "owned_by_humans"}
            ],
            actions=[{"subject": "detected_object", "predicate": "is_a", "object": "pet"},
                    {"subject": "image", "predicate": "contains", "object": "pet"}]
        ))
    
    def load_rules(self, rules_file: str):
        """
        Load rules from a JSON file.
        
        Args:
            rules_file: Path to JSON file containing rules
        """
        with open(rules_file, 'r') as f:
            rules_data = json.load(f)
            
        for rule_data in rules_data:
            rule = SymbolicRule(**rule_data)
            self.add_rule(rule)
    
    def add_rule(self, rule: SymbolicRule):
        """
        Add a rule to the reasoning engine.
        
        Args:
            rule: Rule to add
        """
        self.rules.append(rule)
    
    def reason(self, facts: Dict[str, Any], max_iterations: int = 10) -> Dict[str, Any]:
        """
        Apply rules to infer new facts.
        
        Args:
            facts: Dictionary of initial facts
            max_iterations: Maximum number of reasoning iterations
            
        Returns:
            Dictionary of facts with inferred facts added
        """
        current_facts = facts.copy()
        
        # Apply rules iteratively until no new facts are inferred or max iterations reached
        for _ in range(max_iterations):
            facts_changed = False
            
            for rule in self.rules:
                new_facts = rule.apply(current_facts)
                
                # Check if new facts were inferred
                if new_facts != current_facts:
                    facts_changed = True
                    current_facts = new_facts
            
            # If no new facts were inferred, stop
            if not facts_changed:
                break
        
        return current_facts
    
    def update_knowledge_graph(self, facts: Dict[str, Any]):
        """
        Update the knowledge graph with facts.
        
        Args:
            facts: Dictionary of facts to add to the knowledge graph
        """
        # Add nodes and edges to the knowledge graph
        for subject, properties in facts.items():
            if subject not in self.knowledge_graph:
                self.knowledge_graph.add_node(subject)
                
            if isinstance(properties, dict):
                for predicate, obj in properties.items():
                    if predicate != 'properties' and predicate != 'type':
                        if obj not in self.knowledge_graph:
                            self.knowledge_graph.add_node(obj)
                        self.knowledge_graph.add_edge(subject, obj, relation=predicate)
                
                # Handle type relationship
                if 'type' in properties:
                    obj = properties['type']
                    if obj not in self.knowledge_graph:
                        self.knowledge_graph.add_node(obj)
                    self.knowledge_graph.add_edge(subject, obj, relation='is_a')
                
                # Handle properties
                if 'properties' in properties and isinstance(properties['properties'], list):
                    for prop in properties['properties']:
                        if prop not in self.knowledge_graph:
                            self.knowledge_graph.add_node(prop)
                        self.knowledge_graph.add_edge(subject, prop, relation='has_property')
    
    def get_derived_facts(self, facts: Dict[str, Any]) -> List[str]:
        """
        Get human-readable strings describing derived facts.
        
        Args:
            facts: Dictionary of facts
            
        Returns:
            List of strings describing derived facts
        """
        result = []
        
        # Generate natural language descriptions of facts
        for subject, properties in facts.items():
            if subject == 'image' and isinstance(properties, dict):
                if 'contains' in properties:
                    result.append(f"This image contains a {properties['contains']}.")
            
            if subject == 'detected_object' and isinstance(properties, dict):
                if 'type' in properties:
                    obj_type = properties['type']
                    result.append(f"The detected object is a {obj_type}.")
                    
                    # Add specific conclusions based on type
                    if obj_type == 'pet':
                        result.append("This image contains a pet.")
                
                if 'properties' in properties and isinstance(properties['properties'], list):
                    props = ", ".join(properties['properties'])
                    result.append(f"The detected object has these properties: {props}.")
        
        return result


# Example usage
if __name__ == "__main__":
    # Initialize the reasoning engine
    engine = SymbolicReasoningEngine()
    
    # Define some initial facts
    facts = {
        "detected_object": {
            "type": "cat",
            "properties": ["fluffy", "alive"]
        }
    }
    
    # Apply reasoning
    inferred_facts = engine.reason(facts)
    
    # Update knowledge graph
    engine.update_knowledge_graph(inferred_facts)
    
    # Get derived facts
    derived_facts = engine.get_derived_facts(inferred_facts)
    
    print("Initial facts:", facts)
    print("Inferred facts:", inferred_facts)
    print("Derived facts:", derived_facts) 