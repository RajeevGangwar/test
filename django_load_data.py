import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from django.db import transaction, models
from django.core.validators import MaxValueValidator, MinValueValidator
from concurrent.futures import ThreadPoolExecutor
import logging
import tomlkit
import importlib
from datetime import datetime
from decimal import Decimal

class ModelGenerator:
    """Generates Django models from TOML definitions"""
    
    FIELD_MAPPINGS = {
        'string': models.CharField,
        'text': models.TextField,
        'integer': models.IntegerField,
        'bigint': models.BigIntegerField,
        'float': models.FloatField,
        'decimal': models.DecimalField,
        'boolean': models.BooleanField,
        'date': models.DateField,
        'datetime': models.DateTimeField,
        'time': models.TimeField,
        'json': models.JSONField,
        'foreign_key': models.ForeignKey,
        'one_to_one': models.OneToOneField,
        'many_to_many': models.ManyToManyField,
        'uuid': models.UUIDField,
        'email': models.EmailField,
        'ip': models.GenericIPAddressField,
        'url': models.URLField,
        'binary': models.BinaryField,
    }

    @classmethod
    def generate_model(cls, model_name: str, fields_config: dict) -> type:
        """Generate Django model class from TOML configuration"""
        
        class Meta:
            pass
        
        # Process meta options
        if 'meta' in fields_config:
            meta_config = fields_config.pop('meta')
            for key, value in meta_config.items():
                setattr(Meta, key, value)

        attrs = {
            '__module__': 'django.db.models',
            'Meta': Meta,
            '__str__': lambda self: str(getattr(self, fields_config.get('__str__', 'id')))
        }

        # Process fields
        for field_name, field_config in fields_config.get('fields', {}).items():
            if isinstance(field_config, dict):
                field_type = field_config.pop('type')
                field_class = cls.FIELD_MAPPINGS[field_type]
                
                # Handle validators
                validators = []
                if 'validators' in field_config:
                    validator_config = field_config.pop('validators')
                    if 'min_value' in validator_config:
                        validators.append(MinValueValidator(validator_config['min_value']))
                    if 'max_value' in validator_config:
                        validators.append(MaxValueValidator(validator_config['max_value']))
                    field_config['validators'] = validators
                
                # Handle choices
                if 'choices' in field_config:
                    field_config['choices'] = [
                        (choice['value'], choice['label']) 
                        for choice in field_config['choices']
                    ]
                
                # Handle related models for relationship fields
                if field_type in ['foreign_key', 'one_to_one', 'many_to_many']:
                    field_config['to'] = field_config.pop('related_model')
                    if 'on_delete' in field_config:
                        field_config['on_delete'] = getattr(
                            models, field_config['on_delete']
                        )
                
                attrs[field_name] = field_class(**field_config)
            else:
                # Simple field definition with just type
                field_class = cls.FIELD_MAPPINGS[field_config]
                attrs[field_name] = field_class()

        # Create and return model class
        return type(model_name, (models.Model,), attrs)

class TOMLLoader:
    """Handles loading data into Django models using TOML configuration"""
    
    def __init__(self, connection, toml_path: str, batch_size: int = 5000):
        """
        Initialize loader with TOML configuration
        
        Args:
            connection: Database connection object
            toml_path: Path to TOML configuration file
            batch_size: Batch size for data loading
        """
        self.connection = connection
        self.toml_path = Path(toml_path)
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
        # Generate model
        self.model = self._generate_model()

    def _load_config(self) -> dict:
        """Load TOML configuration"""
        if not self.toml_path.exists():
            raise FileNotFoundError(f"TOML file not found: {self.toml_path}")
            
        with open(self.toml_path, 'r') as f:
            return tomlkit.load(f)

    def _generate_model(self) -> type:
        """Generate Django model from TOML configuration"""
        model_config = self.config.get('model')
        if not model_config:
            raise ValueError("TOML must contain a 'model' section")
            
        model_name = model_config.get('name')
        if not model_name:
            raise ValueError("Model must have a name")
            
        return ModelGenerator.generate_model(model_name, model_config)

    def _get_query(self) -> str:
        """Get SQL query from configuration"""
        query_config = self.config.get('query')
        if not query_config:
            raise ValueError("TOML must contain a 'query' section")

        if isinstance(query_config, str):
            # Direct SQL or file path
            if any(keyword in query_config.upper() for keyword in ['SELECT', 'WITH', 'FROM']):
                return query_config
            
            query_path = self.toml_path.parent / query_config
            if not query_path.exists():
                raise FileNotFoundError(f"Query file not found: {query_path}")
            
            with open(query_path, 'r') as f:
                return f.read().strip()
        
        elif isinstance(query_config, dict):
            sql = query_config.get('sql')
            path = query_config.get('path')
            params = query_config.get('params', {})
            
            if sql:
                query = sql
            elif path:
                query_path = self.toml_path.parent / path
                if not query_path.exists():
                    raise FileNotFoundError(f"Query file not found: {query_path}")
                with open(query_path, 'r') as f:
                    query = f.read().strip()
            else:
                raise ValueError("Query must contain either 'sql' or 'path'")
                
            return query.format(**params) if params else query
        
        raise ValueError(f"Invalid query configuration: {query_config}")

    def fetch_data_in_chunks(self, query: str) -> pd.DataFrame:
        """Fetch data from database in chunks"""
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            while True:
                chunk = cursor.fetchmany(self.batch_size)
                if not chunk:
                    break
                yield pd.DataFrame(chunk, columns=[desc[0] for desc in cursor.description])
        finally:
            cursor.close()

    @transaction.atomic
    def bulk_create_objects(self, objects: List[Dict]) -> int:
        """Bulk create objects within a transaction"""
        created = self.model.objects.bulk_create(
            [self.model(**obj) for obj in objects],
            batch_size=self.batch_size,
            ignore_conflicts=True
        )
        return len(created)

    def transform_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations defined in TOML to data chunk"""
        transform_config = self.config.get('transform', {})
        
        for column, transforms in transform_config.items():
            if column not in chunk.columns:
                continue
                
            for transform in transforms:
                if transform == 'strip':
                    chunk[column] = chunk[column].str.strip()
                elif transform == 'lower':
                    chunk[column] = chunk[column].str.lower()
                elif transform == 'upper':
                    chunk[column] = chunk[column].str.upper()
                elif transform == 'title':
                    chunk[column] = chunk[column].str.title()
                elif isinstance(transform, dict):
                    if 'replace' in transform:
                        chunk[column] = chunk[column].replace(
                            transform['replace']['from'],
                            transform['replace']['to']
                        )
                    elif 'custom' in transform:
                        transform_func = self._import_callable(transform['custom'])
                        chunk[column] = chunk[column].apply(transform_func)
        
        return chunk

    def _import_callable(self, path: str):
        """Import callable from path"""
        try:
            module_path, callable_name = path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, callable_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Error importing {path}: {str(e)}")

    def load_data(self) -> int:
        """
        Load data into the model
        
        Returns:
            Number of records loaded
        """
        try:
            self.logger.info(f"Starting data load for {self.model.__name__}")
            
            query = self._get_query()
            total_records = 0
            
            for chunk in self.fetch_data_in_chunks(query):
                # Apply transformations
                if 'transform' in self.config:
                    chunk = self.transform_chunk(chunk)
                
                # Convert to records and bulk create
                records = chunk.to_dict('records')
                created = self.bulk_create_objects(records)
                total_records += created
                
                self.logger.info(
                    f"Loaded {created} records for {self.model.__name__}. "
                    f"Total: {total_records}"
                )
            
            return total_records
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

# Example TOML configuration (user_model.toml):
"""
[model]
name = "User"
__str__ = "email"

[model.meta]
db_table = "auth_user"
verbose_name = "User"
verbose_name_plural = "Users"
ordering = ["-created_at"]

[model.fields.email]
type = "email"
max_length = 255
unique = true

[model.fields.username]
type = "string"
max_length = 150
unique = true
help_text = "Required. 150 characters or fewer."

[model.fields.password]
type = "string"
max_length = 128

[model.fields.is_active]
type = "boolean"
default = true

[model.fields.created_at]
type = "datetime"
auto_now_add = true

[model.fields.updated_at]
type = "datetime"
auto_now = true

[model.fields.role]
type = "string"
max_length = 20
choices = [
    { value = "admin", label = "Administrator" },
    { value = "user", label = "Regular User" },
    { value = "guest", label = "Guest" }
]

[query]
sql = '''
    SELECT 
        email,
        username,
        password,
        is_active,
        role
    FROM legacy_users
    WHERE created_at >= {start_date}
'''
params = { start_date = "2024-01-01" }

[transform]
email = ["lower", "strip"]
username = ["strip"]
role = ["lower", { replace = { from = "administrator", to = "admin" } }]
"""

# Example usage:
"""
from myapp.database import get_impala_connection

conn = get_impala_connection()
loader = TOMLLoader(
    connection=conn,
    toml_path='models/user_model.toml'
)

# Create model and load data
total_loaded = loader.load_data()
"""