class MetadataBuilder:

    def __init__(self):
        pass

    def build(
        self,
        language,
        region,
        century,
        script,
        confidence,
    ):

        return {

            "language": language,

            "region": region,

            "century": century,

            "script_type": script,

            "confidence": confidence,
        }
