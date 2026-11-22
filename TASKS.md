# FabaCare task plan

Tasks below are planned from the weekly progress report. Corn integration is exploratory and depends on the colleague's module being available and its interface being compatible.

## Next up

- [ ] **Evaluate and refine SAM segmentation**
  - Check representative single-leaf and multi-leaf images.
  - Investigate fragmented masks, duplicate regions, incomplete boundaries, and background inclusion.
  - Record findings and adjust filtering/selection as evidence supports.
- [ ] **Improve individual-leaf result presentation**
  - Review extracted leaf crops and diagnosis cards.
  - Make individual detections easier to inspect while retaining the original-image segmentation preview.
- [ ] **Explore corn module integration with colleague**
  - Confirm module readiness and available inference interface.
  - Agree on crop selection and how uploads route to the faba bean or corn pipeline.
- [ ] **Define the shared website and API workflow**
  - Document common input/output formats and endpoint requirements.
  - Keep each crop's model and prediction logic separate behind the shared experience.
- [ ] **Run conditional integration tests**
  - Start only if both modules are ready and interfaces are compatible.
  - Test sample faba bean and corn images and document issues.

## Done this week (from the report)

- ResNet18 inference integrated into FastAPI for four faba bean classes.
- SAM ViT-B candidate segmentation connected to the image workflow.
- Mask filtering and duplicate reduction added.
- Frontend connected to image upload and prediction results, including confidence and uncertainty information.
- Initial end-to-end tests run with sample images.
