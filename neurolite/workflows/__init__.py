"""
Domain-specific workflow coordination for NeuroLite.

This module provides task-specific workflow coordination for different domains
including computer vision, NLP, and tabular data tasks. It ensures consistent
API patterns across all domain-specific implementations.
"""

from .base import BaseWorkflow, WorkflowConfig, WorkflowResult
from .vision import VisionWorkflow
from .nlp import NLPWorkflow
from .tabular import TabularWorkflow
from .factory import WorkflowFactory, create_workflow, get_workflow_factory

__all__ = [
    "BaseWorkflow",
    "WorkflowConfig", 
    "WorkflowResult",
    "VisionWorkflow",
    "NLPWorkflow",
    "TabularWorkflow",
    "WorkflowFactory",
    "create_workflow",
    "get_workflow_factory"
]