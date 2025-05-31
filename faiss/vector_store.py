"""
FAISS-based vector store for building damage analysis standard data
Main vector store implementation for production deployment
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import json
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)


class FAISSStandardDataVectorStore:
    """FAISS-based vector store for building damage criteria data"""

    def __init__(self, criteria_data_path: str = "criteria"):
        """
        Initialize the FAISS vector store with criteria data

        Args:
            criteria_data_path: Path to the criteria data directory
        """
        self.criteria_data_path = Path(criteria_data_path)
        self.faiss_index_path = self.criteria_data_path / "faiss_index"
        self.metadata_path = self.criteria_data_path / "metadata.json"

        # Initialize embeddings with caching
        self.embeddings = self._initialize_embeddings()

        # Initialize FAISS vector store
        self.vectorstore = None
        self.documents_metadata = []

        # Define data categories
        self.data_categories = {
            "main_data": "main_data.xlsx",
            "damage_risk_index": "damage_risk_index/",
            "damage_status_recovery": "damage_status_recovery/",
            "location_scores": "location_scores/",
            "unit_prices": "unit_prices/",
            "labor_costs": "labor_costs/",
            "work_types": "work_types/",
        }

        # Load or build vector store
        self._initialize_vectorstore()

    @st.cache_resource
    def _initialize_embeddings(_self):
        """Initialize embeddings with Streamlit caching"""
        try:
            logger.info("Initializing embeddings model...")

            # Suppress HuggingFace warnings during initialization
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", message=".*use_fast.*")

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    model_kwargs={"device": "cpu"},  # Force CPU for deployment
                    encode_kwargs={
                        "batch_size": 16
                    },  # Remove show_progress_bar to avoid conflict
                )

            logger.info("✅ Embeddings model loaded successfully")
            return embeddings
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {e}")
            return None

    def _initialize_vectorstore(self):
        """Initialize or load existing FAISS vector store"""
        if not self.embeddings:
            logger.warning("Embeddings not available, using fallback mode")
            return

        try:
            # Try to load existing FAISS index
            if self.faiss_index_path.exists() and self.metadata_path.exists():
                logger.info("Loading existing FAISS index...")
                self.vectorstore = FAISS.load_local(
                    str(self.faiss_index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )

                # Load metadata
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.documents_metadata = json.load(f)

                logger.info(
                    f"Loaded FAISS index with {len(self.documents_metadata)} documents"
                )
            else:
                logger.info("Building new FAISS index from criteria data...")
                self._build_vectorstore()

        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            self._build_vectorstore()

    def _build_vectorstore(self):
        """Build FAISS vector store from criteria data files"""
        if not self.embeddings:
            logger.warning("Embeddings not available, skipping vector store build")
            return

        logger.info("Building FAISS vector store from criteria data...")

        documents = []

        # Process main Excel file
        main_excel_path = self.criteria_data_path / "main_data.xlsx"
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
            subdir_path = self.criteria_data_path / subdir
            if subdir_path.exists():
                documents.extend(self._process_directory(subdir_path, subdir))

        if documents:
            try:
                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )

                split_docs = text_splitter.split_documents(documents)

                # Create FAISS vector store
                self.vectorstore = FAISS.from_documents(
                    documents=split_docs,
                    embedding=self.embeddings,
                )

                # Save FAISS index
                self.faiss_index_path.mkdir(exist_ok=True)
                self.vectorstore.save_local(str(self.faiss_index_path))

                # Save metadata
                self.documents_metadata = [doc.metadata for doc in split_docs]
                with open(self.metadata_path, "w", encoding="utf-8") as f:
                    json.dump(self.documents_metadata, f, ensure_ascii=False, indent=2)

                logger.info(
                    f"Built FAISS vector store with {len(split_docs)} document chunks"
                )

            except Exception as e:
                logger.error(f"Failed to build FAISS vector store: {e}")
                self.vectorstore = None
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
                        "category": "main_data",
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
            # Try different engines for problematic Excel files
            try:
                excel_file = pd.ExcelFile(file_path)
            except Exception as e:
                logger.warning(f"Failed to read {file_path} with default engine: {e}")
                try:
                    # Try with openpyxl engine
                    excel_file = pd.ExcelFile(file_path, engine="openpyxl")
                except Exception as e2:
                    logger.warning(f"Failed to read {file_path} with openpyxl: {e2}")
                    # Skip this file if both methods fail
                    return documents

            for sheet_name in excel_file.sheet_names:
                try:
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
                    logger.warning(
                        f"Failed to process sheet {sheet_name} in {file_path}: {e}"
                    )
                    continue

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
        """Search for similar documents using FAISS"""
        if not self.vectorstore or not self.embeddings:
            logger.warning("Vector store not available, returning empty results")
            return []

        try:
            if category:
                # FAISS doesn't support filtering directly, so we'll search more and filter
                results = self.vectorstore.similarity_search(query, k=k * 3)
                # Filter by category
                filtered_results = [
                    doc for doc in results if doc.metadata.get("category") == category
                ]
                return filtered_results[:k]
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


def create_faiss_vector_store(
    criteria_data_path: str = "criteria",
) -> FAISSStandardDataVectorStore:
    """Create and return FAISS vector store instance"""
    try:
        return FAISSStandardDataVectorStore(criteria_data_path)
    except Exception as e:
        logger.error(f"Failed to create FAISS vector store: {e}")
        # Return a minimal vector store instance
        vector_store = FAISSStandardDataVectorStore.__new__(
            FAISSStandardDataVectorStore
        )
        vector_store.criteria_data_path = Path(criteria_data_path)
        vector_store.vectorstore = None
        vector_store.embeddings = None
        vector_store.documents_metadata = []
        vector_store.data_categories = {}
        return vector_store


if __name__ == "__main__":
    # Test FAISS vector store creation
    import logging

    logging.basicConfig(level=logging.INFO)

    vector_store = create_faiss_vector_store()

    # Test search
    results = vector_store.search_similar_documents("균열 피해", k=3)
    for i, doc in enumerate(results):
        print(f"Document {i+1}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)
