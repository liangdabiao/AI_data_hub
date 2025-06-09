from supabase import create_client
from utils.logger import logger

class SupabaseDBClient:
    _instance = None  # Singleton instance

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, url, key):
        if not hasattr(self, "_initialized"):
            self.url = url
            self.key = key
            self.client = None
            self._initialized = True

    def connect(self):
        """Establish a Supabase connection."""
        if not self.client:
            try:
                self.client = create_client(self.url, self.key)
                logger.info("Supabase connection established.")
            except Exception as e:
                logger.error(f"Error connecting to Supabase: {e}")
                raise

    # CRUD Methods
    def create(self, table, data):
        """Insert a row into a table."""
        try:
            self.connect()
            response = self.client.table(table).insert(data).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error in CREATE operation: {e}")
            raise

    def read(self, table, conditions=None):
        """Read rows from a table."""
        try:
            self.connect()
            query = self.client.table(table).select("*")
            if conditions:
                for key, value in conditions.items():
                    query = query.eq(key, value)
            response = query.execute()
            return response.data
        except Exception as e:
            logger.error(f"Error in READ operation: {e}")
            raise

    def update(self, table, data, conditions):
        """Update rows in a table."""
        try:
            self.connect()
            query = self.client.table(table).update(data)
            for key, value in conditions.items():
                query = query.eq(key, value)
            response = query.execute()
            return response.data
        except Exception as e:
            logger.error(f"Error in UPDATE operation: {e}")
            raise

    def delete(self, table, conditions):
        """Delete rows from a table."""
        try:
            self.connect()
            query = self.client.table(table).delete()
            for key, value in conditions.items():
                query = query.eq(key, value)
            response = query.execute()
            return response.data
        except Exception as e:
            logger.error(f"Error in DELETE operation: {e}")
            raise