#!/usr/bin/env python3
"""
Build FAISS index for deployment
Run this script before deployment to pre-build the vector store index
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_faiss_index(standard_data_path: str = "standard"):
    """Build FAISS index from standard data"""

    try:
        from .vector_store import create_faiss_vector_store

        logger.info("Starting FAISS index build...")

        # Create vector store (this will build the index)
        vector_store = create_faiss_vector_store(standard_data_path)

        if vector_store and vector_store.vectorstore:
            logger.info("✅ FAISS index built successfully!")

            # Get some stats
            if hasattr(vector_store, "documents_metadata"):
                doc_count = len(vector_store.documents_metadata)
                logger.info(f"📊 Indexed {doc_count} document chunks")

            # Test search
            logger.info("🔍 Testing search functionality...")
            results = vector_store.search_similar_documents("균열 피해", k=3)
            logger.info(f"✅ Search test successful - found {len(results)} results")

            # Check file sizes
            faiss_path = Path(standard_data_path) / "faiss_index"
            metadata_path = Path(standard_data_path) / "metadata.json"

            if faiss_path.exists():
                size_mb = sum(
                    f.stat().st_size for f in faiss_path.rglob("*") if f.is_file()
                ) / (1024 * 1024)
                logger.info(f"📁 FAISS index size: {size_mb:.2f} MB")

            if metadata_path.exists():
                size_kb = metadata_path.stat().st_size / 1024
                logger.info(f"📄 Metadata size: {size_kb:.2f} KB")

            return True

        else:
            logger.error("❌ Failed to build FAISS index")
            return False

    except Exception as e:
        logger.error(f"❌ Error building FAISS index: {e}")
        return False


def verify_standard_data(standard_data_path: str = "standard"):
    """Verify that standard data exists"""

    data_path = Path(standard_data_path)

    if not data_path.exists():
        logger.error(f"❌ Standard data directory not found: {data_path}")
        return False

    # Check for main data file
    main_data = data_path / "main_data.xlsx"
    if not main_data.exists():
        logger.warning(f"⚠️ Main data file not found: {main_data}")
    else:
        size_mb = main_data.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Main data file found: {size_mb:.2f} MB")

    # Check subdirectories
    subdirs = [
        "damage_risk_index",
        "damage_status_recovery",
        "location_scores",
        "unit_prices",
        "labor_costs",
        "work_types",
    ]

    found_dirs = 0
    for subdir in subdirs:
        subdir_path = data_path / subdir
        if subdir_path.exists():
            file_count = len(list(subdir_path.rglob("*.*")))
            logger.info(f"✅ {subdir}: {file_count} files")
            found_dirs += 1
        else:
            logger.warning(f"⚠️ {subdir}: directory not found")

    logger.info(f"📊 Found {found_dirs}/{len(subdirs)} data directories")
    return found_dirs > 0


def clean_old_indexes(standard_data_path: str = "standard"):
    """Clean old vector store indexes"""

    data_path = Path(standard_data_path)

    # Clean FAISS index
    faiss_path = data_path / "faiss_index"
    if faiss_path.exists():
        import shutil

        shutil.rmtree(faiss_path)
        logger.info("🧹 Cleaned old FAISS index")

    # Clean metadata
    metadata_path = data_path / "metadata.json"
    if metadata_path.exists():
        metadata_path.unlink()
        logger.info("🧹 Cleaned old metadata")

    # Clean ChromaDB files (optional)
    chroma_db = data_path / "chroma.sqlite3"
    if chroma_db.exists():
        logger.info(
            f"📋 ChromaDB file exists: {chroma_db.stat().st_size / (1024*1024):.2f} MB"
        )


def main():
    """Main function"""

    logger.info("� FAISS Index Builder for Deployment")
    logger.info("=" * 50)

    # Parse command line arguments
    standard_data_path = "standard"
    if len(sys.argv) > 1:
        standard_data_path = sys.argv[1]

    logger.info(f"📁 Using standard data path: {standard_data_path}")

    # Step 1: Verify standard data
    logger.info("\n📋 Step 1: Verifying standard data...")
    if not verify_standard_data(standard_data_path):
        logger.error("❌ Standard data verification failed")
        sys.exit(1)

    # Step 2: Clean old indexes
    logger.info("\n🧹 Step 2: Cleaning old indexes...")
    clean_old_indexes(standard_data_path)

    # Step 3: Build FAISS index
    logger.info("\n🔨 Step 3: Building FAISS index...")
    if build_faiss_index(standard_data_path):
        logger.info("\n✅ FAISS index build completed successfully!")
        logger.info("🚀 Ready for deployment!")
    else:
        logger.error("\n❌ FAISS index build failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
