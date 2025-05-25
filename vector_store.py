"""
Vector store for building damage analysis standard data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredExcelLoader,
    DirectoryLoader,
    TextLoader,
)
from langchain.schema import Document
import chromadb
from chromadb.config import Settings
import json
import re

logger = logging.getLogger(__name__)


class StandardDataVectorStore:
    """Vector store for standard building damage data"""

    def __init__(self, standard_data_path: str = "standard"):
        """
        Initialize the vector store with standard data

        Args:
            standard_data_path: Path to the standard data directory
        """
        self.standard_data_path = Path(standard_data_path)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Initialize Chroma vector store
        self.vectorstore = None
        self.collection_name = "building_damage_standards"

        # Define data categories with English folder names
        self.data_categories = {
            "main_data": "main_data.xlsx",
            "damage_risk_index": "damage_risk_index/",
            "damage_status_recovery": "damage_status_recovery/",
            "location_scores": "location_scores/",
            "unit_prices": "unit_prices/",
            "labor_costs": "labor_costs/",
            "work_types": "work_types/",
        }

        # Load and process all standard data
        self._load_standard_data()

    def _load_standard_data(self):
        """Load and process all standard data"""
        self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        """Initialize or load existing vector store"""
        try:
            # Try to load existing vector store
            self.vectorstore = Chroma(
                persist_directory=str(self.standard_data_path),
                embedding_function=self.embeddings,
            )

            # Check if vector store has data
            if len(self.vectorstore.get()["ids"]) == 0:
                logger.info("Vector store is empty, building from standard data...")
                self._build_vectorstore()
            else:
                logger.info(
                    f"Loaded existing vector store with {len(self.vectorstore.get()['ids'])} documents"
                )

        except Exception as e:
            logger.warning(f"Could not load existing vector store: {e}")
            self._build_vectorstore()

    def _build_vectorstore(self):
        """Build vector store from standard data files"""
        logger.info("Building vector store from standard data...")

        documents = []

        # Process main Excel file
        main_excel_path = self.standard_data_path / "main_data.xlsx"
        if main_excel_path.exists():
            documents.extend(self._process_main_excel(main_excel_path))

        # Process subdirectories
        subdirs = [
            "damage_risk_index",
            "damage_status_recovery",
            "location_scores",
            "unit_prices",
            "labor_costs",
            "work_types",
        ]

        for subdir in subdirs:
            subdir_path = self.standard_data_path / subdir
            if subdir_path.exists():
                documents.extend(self._process_directory(subdir_path, subdir))

        if documents:
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )

            split_docs = text_splitter.split_documents(documents)

            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=str(self.standard_data_path),
            )

            self.vectorstore.persist()
            logger.info(f"Built vector store with {len(split_docs)} document chunks")
        else:
            logger.warning("No documents found to build vector store")

    def _process_main_excel(self, excel_path: Path) -> List[Document]:
        """Process main Excel file"""
        documents = []

        try:
            # Read all sheets
            excel_file = pd.ExcelFile(excel_path)

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)

                # Convert DataFrame to text
                content = f"시트명: {sheet_name}\n\n"
                content += df.to_string(index=False)

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(excel_path),
                        "sheet_name": sheet_name,
                        "type": "main_data",
                    },
                )
                documents.append(doc)

        except Exception as e:
            logger.error(f"Error processing main Excel file: {e}")

        return documents

    def _process_directory(self, dir_path: Path, category: str) -> List[Document]:
        """Process files in a directory"""
        documents = []

        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                try:
                    if file_path.suffix.lower() in [".xlsx", ".xls"]:
                        docs = self._process_excel_file(file_path, category)
                        documents.extend(docs)
                    elif file_path.suffix.lower() in [".txt", ".md"]:
                        docs = self._process_text_file(file_path, category)
                        documents.extend(docs)

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")

        return documents

    def _process_excel_file(self, file_path: Path, category: str) -> List[Document]:
        """Process Excel file"""
        documents = []

        try:
            excel_file = pd.ExcelFile(file_path)

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)

                content = f"카테고리: {category}\n"
                content += f"파일명: {file_path.name}\n"
                content += f"시트명: {sheet_name}\n\n"
                content += df.to_string(index=False)

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "category": category,
                        "sheet_name": sheet_name,
                        "type": "standard_data",
                    },
                )
                documents.append(doc)

        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {e}")

        return documents

    def _process_text_file(self, file_path: Path, category: str) -> List[Document]:
        """Process text file"""
        documents = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            content = f"카테고리: {category}\n"
            content += f"파일명: {file_path.name}\n\n"
            content += content

            doc = Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "category": category,
                    "type": "standard_data",
                },
            )
            documents.append(doc)

        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")

        return documents

    def search_similar_documents(
        self, query: str, k: int = 5, category: str = None
    ) -> List[Document]:
        """Search for similar documents"""
        if not self.vectorstore:
            return []

        try:
            if category:
                # Filter by category
                results = self.vectorstore.similarity_search(
                    query, k=k, filter={"category": category}
                )
            else:
                results = self.vectorstore.similarity_search(query, k=k)

            return results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def get_damage_risk_index(self, damage_type: str, severity: int) -> Dict[str, Any]:
        """Get damage risk index from standards"""
        query = f"피해위험지수 {damage_type} 심각도 {severity}"
        docs = self.search_similar_documents(query, k=3, category="damage_risk_index")

        risk_data = {
            "risk_index": 3,  # Default medium risk
            "description": "표준 위험도",
            "references": [],
        }

        for doc in docs:
            risk_data["references"].append(
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", ""),
                }
            )

        return risk_data

    def get_repair_standards(self, damage_type: str, area: float) -> Dict[str, Any]:
        """Get repair standards and cost estimates"""
        query = f"복구 근거 {damage_type} 면적 {area}"
        docs = self.search_similar_documents(
            query, k=5, category="damage_status_recovery"
        )

        repair_data = {
            "repair_basis": "표준 복구 방법",
            "work_type": "일반 보수",
            "materials": [],
            "labor": {},
            "duration_days": 1,
            "cost_estimate": 0,
            "priority_score": 50,
            "references": [],
        }

        # Extract information from documents
        for doc in docs:
            content = doc.page_content.lower()

            # Extract work types
            if "공종" in content:
                repair_data["work_type"] = self._extract_work_type(doc.page_content)

            # Extract materials
            materials = self._extract_materials(doc.page_content)
            repair_data["materials"].extend(materials)

            repair_data["references"].append(
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", ""),
                }
            )

        # Get cost estimates
        cost_data = self.get_cost_estimates(damage_type, area)
        repair_data.update(cost_data)

        return repair_data

    def get_cost_estimates(self, damage_type: str, area: float) -> Dict[str, Any]:
        """Get cost estimates from standards"""
        # Search for unit costs
        unit_cost_query = f"단가 {damage_type}"
        unit_docs = self.search_similar_documents(
            unit_cost_query, k=3, category="unit_prices"
        )

        # Search for labor costs
        labor_query = f"노무비 {damage_type}"
        labor_docs = self.search_similar_documents(
            labor_query, k=3, category="labor_costs"
        )

        cost_data = {
            "material_cost": 0,
            "labor_cost": 0,
            "total_cost": 0,
            "unit_cost": 0,
            "labor_days": 1,
            "workers_per_day": 1,
        }

        # Extract cost information (simplified)
        base_unit_cost = 50000  # Default unit cost per m²
        base_labor_cost = 200000  # Default labor cost per day

        cost_data["unit_cost"] = base_unit_cost
        cost_data["material_cost"] = base_unit_cost * area
        cost_data["labor_cost"] = base_labor_cost * cost_data["labor_days"]
        cost_data["total_cost"] = cost_data["material_cost"] + cost_data["labor_cost"]

        return cost_data

    def _extract_work_type(self, content: str) -> str:
        """Extract work type from content"""
        work_types = [
            "도장공사",
            "방수공사",
            "타일공사",
            "석공사",
            "목공사",
            "철근콘크리트공사",
            "조적공사",
            "미장공사",
            "지붕공사",
        ]

        for work_type in work_types:
            if work_type in content:
                return work_type

        return "일반 보수공사"

    def _extract_materials(self, content: str) -> List[str]:
        """Extract materials from content"""
        materials = []

        material_keywords = [
            "시멘트",
            "모르타르",
            "콘크리트",
            "철근",
            "타일",
            "페인트",
            "방수재",
            "실리콘",
            "우레탄",
            "에폭시",
            "석재",
            "목재",
        ]

        for keyword in material_keywords:
            if keyword in content:
                materials.append(keyword)

        return list(set(materials))  # Remove duplicates

    def calculate_priority_score(self, damage_data: Dict[str, Any], area: float) -> int:
        """Calculate repair priority score"""
        base_score = 50

        # Adjust based on severity
        severity = damage_data.get("severity_level", 3)
        severity_multiplier = {1: 0.5, 2: 0.7, 3: 1.0, 4: 1.3, 5: 1.5}

        # Adjust based on area
        area_factor = min(area / 100, 2.0)  # Cap at 2x for large areas

        # Adjust based on damage types
        critical_damages = ["구조적 변형", "기초 침하", "화재 손상"]
        damage_types = damage_data.get("damage_types", [])

        critical_factor = 1.0
        for damage_type in damage_types:
            if any(critical in damage_type for critical in critical_damages):
                critical_factor = 1.5
                break

        priority_score = int(
            base_score
            * severity_multiplier.get(severity, 1.0)
            * area_factor
            * critical_factor
        )
        return min(priority_score, 100)  # Cap at 100

    def _load_documents(self) -> List[Document]:
        """Load documents from standard data directory"""
        documents = []

        try:
            # Load main data file
            main_data_file = self.standard_data_path / "main_data.xlsx"
            if main_data_file.exists():
                df = pd.read_excel(main_data_file)
                for idx, row in df.iterrows():
                    content = " ".join(
                        [str(val) for val in row.values if pd.notna(val)]
                    )
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": "main_data.xlsx",
                            "category": "main_data",
                            "row_index": idx,
                        },
                    )
                    documents.append(doc)

            # Load documents from each category folder
            for category, folder_path in self.data_categories.items():
                if category == "main_data":
                    continue

                category_path = self.standard_data_path / folder_path
                if category_path.exists() and category_path.is_dir():
                    documents.extend(
                        self._load_category_documents(category_path, category)
                    )

            logger.info(f"Loaded {len(documents)} documents from standard data")
            return documents

        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []


def create_vector_store(
    standard_data_path: str = "standard",
) -> StandardDataVectorStore:
    """Create and return vector store instance"""
    return StandardDataVectorStore(standard_data_path)


if __name__ == "__main__":
    # Test vector store creation
    import logging

    logging.basicConfig(level=logging.INFO)

    vector_store = create_vector_store()

    # Test search
    results = vector_store.search_similar_documents("균열 피해", k=3)
    for i, doc in enumerate(results):
        print(f"Document {i+1}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)
