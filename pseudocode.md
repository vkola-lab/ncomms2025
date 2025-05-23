# Pseudocode for model framework.

```
FUNCTION two_stage_training():
    // Load pretrained model from 13-label classification task
    pretrained_model = load_pretrained_transformer("previous_model_checkpoint")
    
    // STAGE 1: Fine-tune for Aβ and meta-τ prediction
    STAGE_1():
        // Transfer weights from pretrained model
        new_model.transformer_encoder = pretrained_model.transformer_encoder
        new_model.modality_embeddings = extract_overlapping_embeddings(pretrained_model)
        
        // Initialize new binary classification heads
        new_model.binary_heads = {
            "Ab": initialize_binary_head(),
            "meta_tau": initialize_binary_head()
        }
        
        // Phase 1a: Train only newly initialized weights (15 epochs)
        freeze_parameters(new_model.transformer_encoder)
        freeze_parameters(new_model.modality_embeddings)
        
        FOR epoch IN range(15):
            loss = train_epoch(new_model, train_data, train_new_weights_only=True)
            
        // Phase 1b: Unfreeze transferred weights and train all parameters
        unfreeze_parameters(new_model.transformer_encoder)
        unfreeze_parameters(new_model.modality_embeddings)
        
        WHILE NOT converged:
            loss = train_epoch(new_model, train_data, train_all_weights=True)
    
    // STAGE 2: Further fine-tune for regional τ prediction
    STAGE_2():
        // Add regional τ binary head
        new_model.binary_heads["regional_tau"] = initialize_binary_head()
        
        WHILE NOT converged:
            loss = train_epoch(new_model, regional_tau_data, train_all_weights=True)
    
    RETURN new_model
END FUNCTION
```
