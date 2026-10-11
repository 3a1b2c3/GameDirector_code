from .operators import (
    ImageCropAndResize, LoadGIF, LoadImage, LoadVideo, RouteByExtensionName,
    RouteByType, ToAbsolutePath, ToList,
)


def default_video_operator(
    base_path="",
    max_pixels=1920*1080, height=None, width=None,
    height_division_factor=16, width_division_factor=16,
    num_frames=81, time_division_factor=4, time_division_remainder=1,
    frame_rate=24, fix_frame_rate=False,
):
    return RouteByType(operator_map=[
        (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
            (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
            (("gif",), LoadGIF(
                num_frames, time_division_factor, time_division_remainder,
                frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
            )),
            (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                num_frames, time_division_factor, time_division_remainder,
                frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                frame_rate=frame_rate, fix_frame_rate=fix_frame_rate,
            )),
        ])),
    ])
