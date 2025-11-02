"""
Document ingestion service.
Handles file uploads, loading, and validation.
"""

import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader


class DocumentIngestion:
    """
    Service for handling document ingestion.
    Supports PDF and TXT file formats.
    """
    
    SUPPORTED_EXTENSIONS = ['pdf', 'txt']
    
    @staticmethod
    def validate_file(filename):
        """
        Validate if the file type is supported.
        
        Args:
            filename (str): Name of the file to validate
            
        Returns:
            tuple: (is_valid, file_extension)
        """
        file_extension = filename.split('.')[-1].lower()
        is_valid = file_extension in DocumentIngestion.SUPPORTED_EXTENSIONS
        return is_valid, file_extension
    
    @staticmethod
    def load_document(uploaded_file):
        """
        Load a single document from an uploaded file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            tuple: (success, documents or error_message)
        """
        is_valid, file_extension = DocumentIngestion.validate_file(uploaded_file.name)
        
        if not is_valid:
            return False, f"Unsupported file type: {uploaded_file.name}"
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            if file_extension == 'pdf':
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
            elif file_extension == 'txt':
                loader = TextLoader(tmp_file_path, encoding='utf-8')
                docs = loader.load()
            else:
                return False, f"Unsupported file type: {uploaded_file.name}"
            
            return True, docs
            
        except Exception as e:
            return False, f"Error loading {uploaded_file.name}: {str(e)}"
            
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    @staticmethod
    def load_documents(uploaded_files):
        """
        Load multiple documents from uploaded files.
        
        Args:
            uploaded_files (list): List of Streamlit uploaded file objects
            
        Returns:
            tuple: (all_documents, error_messages)
        """
        all_docs = []
        error_messages = []
        
        for uploaded_file in uploaded_files:
            success, result = DocumentIngestion.load_document(uploaded_file)
            
            if success:
                all_docs.extend(result)
            else:
                error_messages.append(result)
        
        return all_docs, error_messages
    
    @staticmethod
    def get_new_files(current_files, previous_file_names):
        """
        Identify new files that haven't been processed yet.
        
        Args:
            current_files (list): Current list of uploaded files
            previous_file_names (list): List of previously processed file names
            
        Returns:
            tuple: (new_file_names, files_to_process)
        """
        current_file_names = [f.name for f in current_files]
        new_file_names = [name for name in current_file_names if name not in previous_file_names]
        files_to_process = [f for f in current_files if f.name in new_file_names] if new_file_names else current_files
        
        return new_file_names, files_to_process
