#!/usr/bin/env python3
"""
Migration Script: Export data from complex /src implementation to simplified format
Converts TrainingPair and KnowledgeBaseItem to unified KnowledgeItem format
"""

import json
import pathlib
import logging
from typing import List, Dict, Any
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def migrate_training_pairs() -> List[Dict[str, Any]]:
    """Migrate existing training pairs to new format"""
    migrated_items = []
    
    # Look for existing training data
    models_dir = pathlib.Path("breakfix-kb-model/all-mpnet-base-v2/fine-tuned-runs")
    if not models_dir.exists():
        models_dir = pathlib.Path("models/fine-tuned-runs")
    
    if models_dir.exists():
        pairs_files = sorted(models_dir.glob("*/pairs.json"), 
                           key=lambda p: p.parent.name, reverse=True)
        
        for pairs_file in pairs_files:
            try:
                with open(pairs_file, "r", encoding="utf-8") as f:
                    raw_pairs = json.load(f)
                
                for pair in raw_pairs:
                    if "input" in pair and "target" in pair:
                        # Old TrainingPair format
                        migrated_item = {
                            "problem": pair["input"],
                            "solution": pair["target"],
                            "category": "trained",
                            "source": "migrated_training",
                            "tags": ["migrated", "training_pair"]
                        }
                    elif "problem" in pair and "solution" in pair:
                        # Already new format
                        migrated_item = pair
                    else:
                        logger.warning(f"Unknown pair format: {pair}")
                        continue
                    
                    migrated_items.append(migrated_item)
                
                logger.info(f"Migrated {len(raw_pairs)} items from {pairs_file}")
                
            except Exception as e:
                logger.error(f"Failed to migrate {pairs_file}: {e}")
    
    return migrated_items

def migrate_knowledge_base() -> List[Dict[str, Any]]:
    """Migrate existing knowledge base items to new format"""
    migrated_items = []
    
    try:
        # Try to import from existing src structure
        import sys
        sys.path.append("src")
        
        try:
            from data.knowledge_base import knowledge_base
            
            for item in knowledge_base:
                if hasattr(item, 'description') and hasattr(item, 'resolution'):
                    # Old KnowledgeBaseItem format
                    migrated_item = {
                        "problem": item.description,
                        "solution": item.resolution,
                        "category": "general",
                        "source": "migrated_kb",
                        "tags": ["migrated", "knowledge_base"]
                    }
                    migrated_items.append(migrated_item)
            
            logger.info(f"Migrated {len(migrated_items)} knowledge base items")
            
        except ImportError:
            logger.warning("Could not import existing knowledge base")
            
    except Exception as e:
        logger.error(f"Failed to migrate knowledge base: {e}")
    
    return migrated_items

def export_to_new_format():
    """Export all data to new simplified format"""
    try:
        # Migrate training pairs
        training_items = migrate_training_pairs()
        
        # Migrate knowledge base
        kb_items = migrate_knowledge_base()
        
        # Combine all items
        all_items = training_items + kb_items
        
        if not all_items:
            logger.warning("No data found to migrate")
            return
        
        # Create export directory
        export_dir = pathlib.Path("migrated_data")
        export_dir.mkdir(exist_ok=True)
        
        # Export as training data format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = export_dir / f"migrated_knowledge_{timestamp}.json"
        
        export_data = {
            "data": all_items,
            "migration_info": {
                "timestamp": datetime.now().isoformat(),
                "total_items": len(all_items),
                "training_items": len(training_items),
                "knowledge_base_items": len(kb_items),
                "format_version": "3.0.0"
            }
        }
        
        with open(export_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Migration complete!")
        logger.info(f"📁 Exported {len(all_items)} items to: {export_file}")
        logger.info(f"📊 Training items: {len(training_items)}")
        logger.info(f"📊 Knowledge base items: {len(kb_items)}")
        
        # Create import instructions
        instructions_file = export_dir / "import_instructions.md"
        with open(instructions_file, "w", encoding="utf-8") as f:
            f.write(f"""# Data Migration Instructions

## Migration Summary
- **Total items migrated**: {len(all_items)}
- **Training items**: {len(training_items)}
- **Knowledge base items**: {len(kb_items)}
- **Export file**: `{export_file.name}`

## How to Import into New System

### Option 1: Training Data Import
```bash
curl -X POST "http://localhost:8080/train" \\
     -H "Content-Type: application/json" \\
     -d @{export_file.name}
```

### Option 2: Python Script Import
```python
import requests
import json

# Load migrated data
with open("{export_file.name}", "r") as f:
    data = json.load(f)

# Import as training data
response = requests.post("http://localhost:8080/train", json=data)
print(response.json())
```

### Option 3: Document Import (for large content)
If you have large documents, convert them to Document format:
```python
documents = [{{
    "title": "Migrated Knowledge",
    "content": "\\n\\n".join([f"Q: {{item['problem']}}\\nA: {{item['solution']}}" for item in data["data"]]),
    "category": "migrated",
    "source": "migration"
}}]

response = requests.post("http://localhost:8080/documents/bulk", json=documents)
```

## Verification
After import, verify with:
```bash
curl -X POST "http://localhost:8080/troubleshoot" \\
     -H "Content-Type: application/json" \\
     -d '{{"text": "test query"}}'
```
""")
        
        logger.info(f"📋 Import instructions saved to: {instructions_file}")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    logger.info("🚀 Starting data migration...")
    export_to_new_format()
    logger.info("✅ Migration script completed!") 