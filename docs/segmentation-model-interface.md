# Segmentation Model Interface

## Purpose

SurfWatch needs a stable contract between segmentation models and the baseline/MARSP video workflows. The contract lets the project keep motion compensation, temporal aggregation, comparison rendering, and metrics code independent from any one model implementation.

The first implementation is the existing SegFormer baseline. Future models should implement the same adapter responsibilities before they are connected to MARSP.

## Adapter Contract

Segmentation adapters live under `src/models/` and follow the `SegmentationModelAdapter` protocol in `src/models/segmentation_interface.py`.

Runtime code should create adapters through `build_segmentation_adapter()` in `src/models/adapter_factory.py`. User-facing model metadata is registered in `src/models/model_registry.py`, so adding a second model should not require branching inside the MARSP, backend, or frontend comparison flows.

Each adapter should provide:

* `metadata()` - returns model name, input size, threshold, device, and output format.
* `preprocess_frame(frame)` - converts a source video frame into model-ready input.
* `predict_probability_map(frame)` - returns a rip-current probability map aligned to the source frame.
* `probability_to_mask(probability_map)` - thresholds probabilities into a binary mask.
* `predict(frame)` - runs the full single-frame prediction flow and returns both outputs.

## Input Format

The standard frame input is an OpenCV video frame:

* shape: `(height, width, 3)`
* dtype: `uint8`
* colour order: BGR
* coordinate space: original source frame

Adapters may resize internally for model inference, but returned outputs must be resized back to the source frame size.

## Output Format

Each prediction returns `SegmentationResult`:

* `probability_map`: 2D `float32` NumPy array aligned to the source frame, with values expected in `[0.0, 1.0]`.
* `binary_mask`: 2D `uint8` NumPy array aligned to the source frame, with values `0` for background and `1` for rip-current pixels.
* `metadata`: `SegmentationModelMetadata` describing the adapter configuration used for the prediction.

This keeps probability video writing, mask video writing, overlay rendering, temporal aggregation, and stability metrics consistent across models.

## Thresholds

Mask thresholds must be adapter configuration, not hard-coded inside the model implementation. The current video segmentation CLI passes `--threshold` into `SegFormerSegmentationAdapter`, and the adapter uses that value in `probability_to_mask()`.

## Error Handling

Adapters should raise `ValueError` for invalid frame input, invalid probability map shape, and invalid configuration such as non-positive input sizes. Model checkpoint load errors should surface during adapter initialisation so pipeline failures happen before video processing starts.

## Current SegFormer Adapter

`src/models/segformer_adapter.py` maps the existing SegFormer workflow into the interface:

* `BaselineConfig.pretrained_model_name` selects the Hugging Face SegFormer checkpoint family.
* `BaselineConfig.image_size` controls inference resize dimensions.
* The trained checkpoint is loaded from the existing `model_state_dict`.
* BGR frames are resized, converted to RGB, converted to float tensors, and normalised to `[0.0, 1.0]`.
* The rip-current class probability is read from class index `1`.
* Probability maps are resized back to the original frame size before thresholding.

The command-line video segmentation runner selects this adapter with `--model segformer`, which is currently the default and only supported model.

## Runtime Model Selection

The comparison API exposes `GET /api/models`, returning the default model and every runnable adapter registered by the backend. The Analyse page uses that response to populate its model selector and submits the selected model with `POST /api/comparisons`.

The backend validates the submitted model before starting a job, records it in job and case responses, and passes it to `run_baseline_vs_marsp_compare.py` with `--model`. The comparison script then uses the same adapter for both the baseline and MARSP branches, preserving a fair comparison.

Register a model only when its adapter and compatible trained checkpoint are available. A research candidate should not appear in the runtime registry merely because its architecture is planned.

## Verification

When a new adapter is added, verify that:

* A sample BGR frame produces a 2D probability map matching the original frame size.
* A sample probability map produces a `uint8` binary mask with only `0` and `1` values.
* The baseline and MARSP workflows can consume the adapter output without model-specific branching.
* Output overlay, mask, and probability videos remain compatible with the frontend comparison flow.
