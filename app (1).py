import cv2
import gradio as gr
import json
import numpy as np
import spaces
import torch
import torch.nn as nn

from einops import rearrange
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from skimage.exposure import match_histograms
from transformers import AutoModel


class ModelForGradCAM(nn.Module):
    def __init__(self, model, female):
        super().__init__()
        self.model = model
        self.female = female

    def forward(self, x):
        return self.model(x, self.female, return_logits=True)


def convert_bone_age_to_string(bone_age: float):
    # bone_age in months
    years = round(bone_age // 12)
    months = bone_age - (years * 12)
    months = round(months)

    if months == 12:
        years += 1
        months = 0

    if years == 0:
        str_output = f"{months} months" if months != 1 else "1 month"
    else:
        if months == 0:
            str_output = f"{years} years" if years != 1 else "1 year"
        else:
            str_output = (
                f"{years} years, {months} months"
                if months != 1
                else f"{years} years, 1 month"
            )
    return str_output


@spaces.GPU
def predict_bone_age(Radiograph, Sex, Heatmap):
    x = crop_model.preprocess(Radiograph)
    x = torch.from_numpy(x).float().to(device)
    x = rearrange(x, "h w -> 1 1 h w")
    # crop
    img_shape = torch.tensor([Radiograph.shape[:2]]).to(device)
    with torch.inference_mode():
        box = crop_model(x, img_shape=img_shape).to("cpu").numpy()
    x, y, w, h = box[0]
    cropped = Radiograph[y : y + h, x : x + w]
    # histogram matching
    x = match_histograms(cropped, ref_img)

    x = model.preprocess(x)
    x = torch.from_numpy(x).float().to(device)
    x = rearrange(x, "h w -> 1 1 h w")
    female = torch.tensor([Sex]).to(device)
    with torch.inference_mode():
        bone_age = model(x, female)[0].item()

    # get closest G&P ages
    # from: https://rad.esmil.com/Reference/G_P_BoneAge/
    gp_ages = greulich_and_pyle_ages["female" if Sex else "male"]
    diffs_gp = np.abs(bone_age - gp_ages)
    diffs_gp = np.argsort(diffs_gp)
    closest1 = gp_ages[diffs_gp[0]]
    closest2 = gp_ages[diffs_gp[1]]

    bone_age_str = convert_bone_age_to_string(bone_age)
    closest1 = convert_bone_age_to_string(closest1)
    closest2 = convert_bone_age_to_string(closest2)

    if Heatmap:
        # net1 and net2 to give good GradCAMs
        # net0 is bad for some reason
        # because GradCAM expects 1 input tensor, need to
        # pass female during class instantiation
        model_grad_cam = ModelForGradCAM(model.net1, female)
        target_layers = [model_grad_cam.model.backbone.stages[-1]]
        targets = [ClassifierOutputTarget(round(bone_age))]
        with GradCAM(model=model_grad_cam, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=x, targets=targets, eigen_smooth=True)

        heatmap = cv2.applyColorMap(
            (grayscale_cam[0] * 255).astype("uint8"), cv2.COLORMAP_JET
        )
        image = cv2.cvtColor(
            x[0, 0].to("cpu").numpy().astype("uint8"), cv2.COLOR_GRAY2RGB
        )
        image_weight = 0.6
        grad_cam_image = (1 - image_weight) * heatmap[..., ::-1] + image_weight * image
        grad_cam_image = grad_cam_image
    else:
        # if no heatmap desired, just show image
        grad_cam_image = cv2.cvtColor(x[0, 0].to("cpu").numpy(), cv2.COLOR_GRAY2RGB)

    return (
        bone_age_str,
        f"The closest Greulich & Pyle bone ages are:\n  1) {closest1}\n  2) {closest2}",
        grad_cam_image.astype("uint8"),
    )


image = gr.Image(image_mode="L")
sex = gr.Radio(["Male", "Female"], type="index")
generate_heatmap = gr.Radio(["No", "Yes"], type="index")
label = gr.Label(show_label=False)
textbox = gr.Textbox(show_label=False)
grad_cam_image = gr.Image(image_mode="RGB", label="Heatmap / Image")

with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Deep Learning Model for Pediatric Bone Age

    This model predicts the bone age from a single frontal view hand radiograph. Read more about the model here:
    <https://huggingface.co/ianpan/bone-age>

    There is also an option to output a heatmap over the radiograph to show regions where the model is focusing on
    to make its prediction. However, this takes extra computation and will increase the runtime.

    This model is for demonstration purposes only and has NOT been approved by any regulatory agency for clinical use. The user assumes
    any and all responsibility regarding their own use of this model and its outputs. Do NOT upload any images containing protected
    health information, as this demonstration is not compliant with patient privacy laws.

    Created by: Ian Pan, <https://ianpan.me>
    
    Last updated: December 16, 2024
    """
    )
    gr.Interface(
        fn=predict_bone_age,
        inputs=[image, sex, generate_heatmap],
        outputs=[label, textbox, grad_cam_image],
        examples=[
            ["examples/2639.png", "Female", "Yes"],
            ["examples/10043.png", "Female", "No"],
            ["examples/8888.png", "Female", "Yes"],
        ],
        cache_examples=True,
    )

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device `{device}` ...")

    crop_model = AutoModel.from_pretrained(
        "ianpan/bone-age-crop", trust_remote_code=True
    )
    model = AutoModel.from_pretrained("ianpan/bone-age", trust_remote_code=True)

    crop_model, model = crop_model.eval().to(device), model.eval().to(device)

    ref_img = cv2.imread("ref_img.png", 0)

    with open("greulich_and_pyle_ages.json", "r") as f:
        greulich_and_pyle_ages = json.load(f)["bone_ages"]

    greulich_and_pyle_ages = {
        k: np.asarray(v) for k, v in greulich_and_pyle_ages.items()
    }

    demo.launch(share=True)
