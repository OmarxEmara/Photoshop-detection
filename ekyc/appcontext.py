from typing import Optional
from redis.asyncio import Redis  
from utils.config import Config
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine
import asyncio
import logging
from utils.mq import AsyncRabbitMQInstance



logger = logging.getLogger(__name__)


class AppContext:
    """
    Manages Global Application state with proper async resource management.
    """

    def __init__(self):
        self.redis: Optional[Redis] = None
        
        
        self.rabbitmq: Optional[AsyncRabbitMQInstance] = None
        

        self.db_engine = None  
        self.async_db_engine = None  
        self.db_session_factory: Optional[sessionmaker] = None
        self.async_db_session_factory: Optional[async_sessionmaker] = None
        
        
        self.gestures = ["left", "right", "blink", "smile", "straight"]
        

    async def initialize(self):
        """
        Initialize all application resources asynchronously.
        """
        config = Config()
        
        try:
            
            await self._initialize_redis(config)
            
            
            await self._initialize_database(config)
            
            
            await self._initialize_rabbitmq()
            
            logger.info("Application context initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize application context: {e}")
            await self.cleanup()
            raise RuntimeError(f"Failed to initialize application context: {str(e)}")

    async def _initialize_redis(self, config: Config):
        """Initialize Redis connection"""
        try:
            self.redis = Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            
            await self.redis.ping()
            self._redis_healthy = True
            logger.info("Redis connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self._redis_healthy = False
            raise

    async def _initialize_database(self, config: Config):
        """Initialize database connections"""
        try:
            self.db_engine = create_engine(
                config.DATABASE_URL, 
                pool_pre_ping=True,
                pool_recycle=3600,
            )
            self.db_session_factory = sessionmaker(bind=self.db_engine)
            
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self._db_healthy = False
            raise

    async def _initialize_rabbitmq(self):
        """Initialize RabbitMQ connection"""
        try:
            self.rabbitmq = AsyncRabbitMQInstance(self.redis)
            await self.rabbitmq.connect()
            
            self._rabbitmq_healthy = True
            logger.info("RabbitMQ connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RabbitMQ: {e}")
            self._rabbitmq_healthy = False
            raise

    async def cleanup(self):
        """
        Cleanup all resources before shutdown.
        """
        logger.info("Starting application context cleanup")
        

        if self.rabbitmq:
            try:
                await self.rabbitmq.disconnect()
                logger.info("RabbitMQ connection closed")
            except Exception as e:
                logger.error(f"Error closing RabbitMQ connection: {e}")
        

        if self.redis:
            try:
                await self.redis.aclose()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
        
        if self.db_engine:
            try:
                self.db_engine.dispose()
                logger.info("Database engine disposed")
            except Exception as e:
                logger.error(f"Error disposing database engine: {e}")
        
        logger.info("Application context cleanup completed")


    def get_db_session(self) -> Session:
        """
        Get a new database session.
        """
        if not self.db_session_factory:
            raise RuntimeError("Database session factory is not initialized.")
        return self.db_session_factory()


    


ctx = AppContext()