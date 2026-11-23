"""Milestone 4: super-resolution."""
import cv2


def upscale(rgb, scale, device="cpu"):
    """Real-ESRGAN if installed (see README), otherwise bicubic with a warning."""
    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        up = RealESRGANer(scale=4, model_path=url, model=net, tile=512, tile_pad=10, pre_pad=0,
                          half=str(device).startswith("cuda") and torch.cuda.is_available())
        out, _ = up.enhance(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), outscale=scale)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] Real-ESRGAN unavailable ({type(e).__name__}: {e}); using bicubic instead.")
        h, w = rgb.shape[:2]
        return cv2.resize(rgb, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

