import models.character_encoder.trainer as character_encoder
import models.classification.trainer as classification


ARCHS = {
    "character_encoder": character_encoder.Model,
    "classification": classification.Model,
}
