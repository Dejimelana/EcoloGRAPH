"""
LM Studio process manager for automatic restart on failure.

Handles detection of corrupted state and clean restart of LM Studio server.
"""
import logging
import os
import subprocess
import time
import psutil
import httpx
from pathlib import Path

logger = logging.getLogger(__name__)


class LMStudioManager:
    """
    Manages LM Studio process lifecycle with health checks and auto-restart.
    
    Features:
    - Process detection and monitoring
    - Health check via API ping
    - Graceful shutdown with fallback to force kill
    - Auto-restart with readiness waiting
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        lm_studio_path: str | None = None,
        startup_timeout: int = 60,
        health_check_timeout: int = 5
    ):
        """
        Initialize LM Studio manager.
        
        Args:
            base_url: LM Studio API base URL
            lm_studio_path: Path to LM Studio executable (auto-detect if None)
            startup_timeout: Max seconds to wait for startup
            health_check_timeout: Timeout for health check requests
        """
        self.base_url = base_url.rstrip("/")
        self.lm_studio_path = lm_studio_path or self._find_lm_studio()
        self.startup_timeout = startup_timeout
        self.health_check_timeout = health_check_timeout
        
        logger.info(f"LMStudioManager initialized (base_url={base_url})")
    
    def is_healthy(self) -> bool:
        """
        Check if LM Studio is running and responsive.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = httpx.get(
                f"{self.base_url}/v1/models",
                timeout=self.health_check_timeout
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def wait_until_ready(self, timeout: int | None = None) -> bool:
        """
        Wait until LM Studio is ready to accept requests.
        
        Args:
            timeout: Max seconds to wait (uses startup_timeout if None)
            
        Returns:
            True if ready, False if timeout
        """
        timeout = timeout or self.startup_timeout
        start_time = time.time()
        
        logger.info(f"Waiting for LM Studio to be ready (timeout={timeout}s)...")
        
        while time.time() - start_time < timeout:
            if self.is_healthy():
                elapsed = time.time() - start_time
                logger.info(f"LM Studio is ready (took {elapsed:.1f}s)")
                return True
            
            time.sleep(2)
        
        logger.error(f"LM Studio did not become ready within {timeout}s")
        return False
    
    def restart(self) -> bool:
        """
        Restart LM Studio server.
        
        Steps:
        1. Stop existing process (graceful → force kill)
        2. Start new process
        3. Wait until ready
        
        Returns:
            True if restart successful, False otherwise
        """
        logger.warning("Restarting LM Studio...")
        
        # Step 1: Stop
        if not self.stop():
            logger.error("Failed to stop LM Studio, attempting restart anyway...")
        
        # Step 2: Start
        if not self.start():
            logger.error("Failed to start LM Studio")
            return False
        
        # Step 3: Wait until ready
        if not self.wait_until_ready():
            logger.error("LM Studio started but did not become ready")
            return False
        
        logger.info("LM Studio restart successful")
        return True
    
    def stop(self, force: bool = False) -> bool:
        """
        Stop LM Studio process.
        
        Args:
            force: If True, kill immediately. If False, try graceful shutdown first.
            
        Returns:
            True if stopped successfully
        """
        processes = self._find_processes()
        
        if not processes:
            logger.info("No LM Studio processes found")
            return True
        
        logger.info(f"Found {len(processes)} LM Studio process(es)")
        
        for proc in processes:
            try:
                if force:
                    logger.warning(f"Force killing process {proc.pid}")
                    proc.kill()
                else:
                    logger.info(f"Terminating process {proc.pid} gracefully")
                    proc.terminate()
                    
                    # Wait up to 10s for graceful shutdown
                    try:
                        proc.wait(timeout=10)
                    except psutil.TimeoutExpired:
                        logger.warning(f"Process {proc.pid} did not terminate, force killing")
                        proc.kill()
                
                logger.info(f"Stopped process {proc.pid}")
            
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.warning(f"Could not stop process {proc.pid}: {e}")
        
        # Wait a bit for cleanup
        time.sleep(2)
        return True
    
    def start(self) -> bool:
        """
        Start LM Studio process.
        
        Returns:
            True if started successfully
        """
        if not self.lm_studio_path:
            logger.error("LM Studio path not found, cannot start")
            return False
        
        if not Path(self.lm_studio_path).exists():
            logger.error(f"LM Studio executable not found: {self.lm_studio_path}")
            return False
        
        try:
            logger.info(f"Starting LM Studio from {self.lm_studio_path}")
            
            # Start detached process
            subprocess.Popen(
                [self.lm_studio_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            )
            
            logger.info("LM Studio process started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start LM Studio: {e}")
            return False
    
    def _find_processes(self) -> list[psutil.Process]:
        """Find all running LM Studio processes."""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'exe']):
            try:
                # Check process name
                name = proc.info['name'].lower()
                exe = (proc.info['exe'] or '').lower()
                
                if 'lmstudio' in name or 'lm studio' in name or 'lmstudio' in exe:
                    processes.append(proc)
            
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return processes
    
    def _find_lm_studio(self) -> str | None:
        """
        Auto-detect LM Studio installation path on Windows.
        
        Returns:
            Path to LM Studio executable, or None if not found
        """
        # Common installation paths on Windows
        potential_paths = [
            os.path.expandvars(r"%LOCALAPPDATA%\LM Studio\LM Studio.exe"),
            os.path.expandvars(r"%PROGRAMFILES%\LM Studio\LM Studio.exe"),
            os.path.expandvars(r"%PROGRAMFILES(X86)%\LM Studio\LM Studio.exe"),
        ]
        
        for path in potential_paths:
            if Path(path).exists():
                logger.info(f"Found LM Studio at: {path}")
                return path
        
        # Try finding via running process
        processes = self._find_processes()
        if processes:
            exe_path = processes[0].exe()
            if exe_path:
                logger.info(f"Found LM Studio via running process: {exe_path}")
                return exe_path
        
        logger.warning("Could not auto-detect LM Studio installation path")
        return None
