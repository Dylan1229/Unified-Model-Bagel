def parse_device_list(value: Optional[str]) :
        if value is None:
            return []
        devices: List[int] = []
        for chunk in value.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            devices.append(int(chunk))
        return devices
    
def get_inferencer(device_id: int):
        instance = inferencer_cache.get(device_id)
        if instance is None:
            if device_id not in vae_cache:
                localized_vae = copy.deepcopy(vae_model).to(f"cuda:{device_id}").eval()
                vae_cache[device_id] = localized_vae
            instance = InterleaveInferencer(
                model=model,
                vae_model=vae_cache[device_id],
                tokenizer=tokenizer,
                vae_transform=vae_transform,
                vit_transform=vit_transform,
                new_token_ids=new_token_ids,
            )
            inferencer_cache[device_id] = instance
        return instance
    
def finalize_text_results(results: List[TextToImageResult]):
        for outcome in results:
            if outcome.error:
                raise RuntimeError(
                    f"Text-to-image task '{outcome.task.task_id}' failed."
                ) from outcome.error
            image = outcome.image
            if image is None:
                raise RuntimeError(f"No image returned for task '{outcome.task.task_id}'")
            image_path = ensure_path(outcome.task.output_image, outcome.task.task_id, output_dir, ".png")
            image.save(image_path)

            thinking_path: Optional[Path] = None
            if outcome.thinking_text:
                thinking_path = ensure_path(
                    outcome.task.output_text,
                    f"{outcome.task.task_id}_thinking",
                    output_dir,
                    ".txt",
                )
                thinking_path.write_text(outcome.thinking_text, encoding="utf-8")

            summary_records.append(
                (
                    outcome.task_index,
                    {
                        "task_id": outcome.task.task_id,
                        "type": outcome.task.kind,
                        "prompt": outcome.task.prompt,
                        "image_path": str(image_path),
                        "thinking_path": str(thinking_path) if thinking_path else None,
                    },
                )
            )