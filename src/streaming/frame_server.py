"""
Frame Server Stub for Virtual Camera / NDI Output.

This is a stub module prepared for future streaming integration.
When implemented, it will provide:
    - Virtual camera output (via pyvirtualcam) for OBS/Zoom/etc.
    - NDI output for professional streaming setups
    - HTTP MJPEG stream for browser-based viewing

Currently, the recommended approach is to use OBS Studio's
Window Capture feature to capture the pygame dashboard window directly.

Required packages for future implementation:
    - pyvirtualcam: Virtual camera for OBS/Zoom
    - python-ndi: NDI output for network streaming

Usage (future):
    server = FrameServer(mode='virtual_camera')
    server.start()

    # In training loop:
    server.send_frame(dashboard.screen)

    server.stop()
"""


class FrameServer:
    """
    Virtual camera / NDI frame server stub.

    Future implementation will support:
    - 'virtual_camera': Create a virtual webcam source
    - 'ndi': Network Device Interface for professional streaming
    - 'mjpeg': HTTP MJPEG stream for browser viewing

    Args:
        mode: Server mode. Options: 'virtual_camera', 'ndi', 'mjpeg'.
        resolution: Output resolution (width, height). Default 1920x1080.
        fps: Output frame rate. Default 30.
    """

    def __init__(
        self,
        mode: str = 'virtual_camera',
        resolution: tuple = (1920, 1080),
        fps: int = 30,
    ):
        self.mode = mode
        self.resolution = resolution
        self.fps = fps
        self.is_running = False

    def start(self) -> None:
        """
        Start the frame server.

        Not yet implemented. Prints a message directing users to
        use OBS Window Capture instead.
        """
        print(f'Frame server ({self.mode}) is not yet implemented.')
        print('For streaming, use OBS Studio with Window Capture on the')
        print('Mario ML Dashboard window. See README.md for setup guide.')

    def send_frame(self, surface) -> None:
        """
        Send a frame to the virtual output.

        Not yet implemented.

        Args:
            surface: pygame.Surface to send.
        """
        pass  # Stub — no-op until implemented

    def stop(self) -> None:
        """
        Stop the frame server.

        Not yet implemented.
        """
        self.is_running = False
