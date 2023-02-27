import html
import time
from typing import List

import streamlit as st
import torch
from annotated_text import annotation
from classy.scripts.cli import import_module_and_submodules
from classy.utils.lightning import (
    load_prediction_dataset_conf_from_checkpoint,
    load_classy_module_from_checkpoint,
)
from omegaconf import OmegaConf

from src.data.data_drivers.base import ExmodSample


@st.cache(allow_output_mutation=True)
def load_model(
    model_checkpoint_path: str,
    prediction_params_path: str,
    cuda_device: int,
    examples: List[ExmodSample],
):

    import_module_and_submodules("classy")

    # load model
    dataset_conf = load_prediction_dataset_conf_from_checkpoint(model_checkpoint_path)
    model = load_classy_module_from_checkpoint(model_checkpoint_path)
    model.to(torch.device(cuda_device if cuda_device != -1 else "cpu"))
    model.freeze()
    if prediction_params_path is not None:
        model.load_prediction_params(dict(OmegaConf.load(prediction_params_path)))

    # mock call to load resources
    list(model.predict([examples[0]], dataset_conf=dataset_conf))

    # return
    return model, dataset_conf


def main(
    model_checkpoint_path: str,
    prediction_params_path: str,
    cuda_device: int,
):

    # setup examples
    examples = [
        ExmodSample(
            source_language="en",
            D=[("dog", "a domestic animal")],
            target_language="en",
        ),
        ExmodSample(
            source_language="en",
            D=[("bank", "sloping land (especially the slope beside a body of water)")],
            target_language="en",
        ),
        ExmodSample(
            source_language="en",
            D=[("dog", "a domestic animal")],
            target_language="en",
        ),
    ]

    # load model
    model, dataset_conf = load_model(
        model_checkpoint_path, prediction_params_path, cuda_device, examples
    )

    # css rules
    st.write(
        """
            <style type="text/css">
                a {
                    text-decoration: none !important;
                }
            </style>
        """,
        unsafe_allow_html=True,
    )

    # setup header
    st.write(
        """
            <div align="center">
                <h1 style='text-align: center;'>
                    <a href="https://sunglasses-ai.github.io/classy/">
                        <img alt="Python" style="height: 1.5em; margin: 0em 0.25em 0em;" src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4NCjwhLS0gR2VuZXJhdG9yOiBBZG9iZSBJbGx1c3RyYXRvciAxNy4wLjAsIFNWRyBFeHBvcnQgUGx1Zy1JbiAuIFNWRyBWZXJzaW9uOiA2LjAwIEJ1aWxkIDApICAtLT4NCjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+DQo8c3ZnIHZlcnNpb249IjEuMSIgaWQ9IkxpdmVsbG9fMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgeD0iMHB4IiB5PSIwcHgiDQoJIHdpZHRoPSI0MzcuNnB4IiBoZWlnaHQ9IjQxMy45NzhweCIgdmlld0JveD0iMCAwIDQzNy42IDQxMy45NzgiIGVuYWJsZS1iYWNrZ3JvdW5kPSJuZXcgMCAwIDQzNy42IDQxMy45NzgiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPGc+DQoJPHBhdGggZmlsbD0iI0ZGQ0MwMCIgZD0iTTM5NC42NjgsMTczLjAzOGMtMS4wMTgsMC0yLjAyLDAuMDY1LTMuMDE1LDAuMTUyQzM3NS42NjUsODIuODExLDI5Ni43OSwxNC4xNDYsMjAxLjgyMywxNC4xNDYNCgkJQzk1LjMxOSwxNC4xNDYsOC45OCwxMDAuNDg1LDguOTgsMjA2Ljk4OWMwLDEwNi41MDUsODYuMzM5LDE5Mi44NDMsMTkyLjg0MywxOTIuODQzYzk0Ljk2NywwLDE3My44NDItNjguNjY1LDE4OS44MjktMTU5LjA0NA0KCQljMC45OTUsMC4wODcsMS45OTcsMC4xNTIsMy4wMTUsMC4xNTJjMTguNzUxLDAsMzMuOTUyLTE1LjIsMzMuOTUyLTMzLjk1MkM0MjguNjIsMTg4LjIzOSw0MTMuNDE5LDE3My4wMzgsMzk0LjY2OCwxNzMuMDM4eg0KCQkgTTIwMS44MjMsMzQ2Ljg2OWMtNzcuMTMsMC0xMzkuODgtNjIuNzUtMTM5Ljg4LTEzOS44OGMwLTc3LjEyOSw2Mi43NS0xMzkuODc5LDEzOS44OC0xMzkuODc5czEzOS44OCw2Mi43NSwxMzkuODgsMTM5Ljg3OQ0KCQlDMzQxLjcwMywyODQuMTE5LDI3OC45NTMsMzQ2Ljg2OSwyMDEuODIzLDM0Ni44Njl6Ii8+DQoJPGc+DQoJCTxwYXRoIGZpbGw9IiNGRkNDMDAiIGQ9Ik0xMTQuOTA3LDIzMy40NzNjLTE0LjYyNiwwLTI2LjQ4My0xMS44NTYtMjYuNDgzLTI2LjQ4M2MwLTYyLjUyOCw1MC44NzEtMTEzLjQwMiwxMTMuMzk4LTExMy40MDINCgkJCWMxNC42MjYsMCwyNi40ODMsMTEuODU2LDI2LjQ4MywyNi40ODNzLTExLjg1NiwyNi40ODMtMjYuNDgzLDI2LjQ4M2MtMzMuMzI0LDAtNjAuNDMzLDI3LjExMi02MC40MzMsNjAuNDM2DQoJCQlDMTQxLjM5LDIyMS42MTcsMTI5LjUzNCwyMzMuNDczLDExNC45MDcsMjMzLjQ3M3oiLz4NCgk8L2c+DQo8L2c+DQo8L3N2Zz4NCg==">
                    </a>
                    Exemplification Modeling
                </h1>
            </div> 
        """,
        unsafe_allow_html=True,
    )

    # how it works
    def hiw():
        st.markdown(
            """
                ## How it works
    
                This space hosts a re-implementation of the original seq2seq approach for Exemplification Modeling, i.e., the task 
                of generating usage examples given one or more words along with their textual definitions, presented by 
                [Barba et al. (2021)](https://www.ijcai.org/proceedings/2021/520).
            """
        )
        st.image("./data/repo-assets/example-image.png", caption="ExtEnD Formulation")
        st.markdown(
            """
            Given the lemma *bank* and its definition *a building where financial services are offered*,
            the model auto-regressively generates the example *He want to the bank to deposit a check*, highlighting
            via a markup notation where *bank* occurs.

            ##### Links:
             * [GitHub](https://www.github.com/poccio/exmod)
             * [Reference Paper](https://www.ijcai.org/proceedings/2021/520)
        """
        )

    # demo
    def demo():

        st.markdown("## Demo")

        # define placeholder
        placeholder = st.selectbox(
            "Examples",
            options=[f"{e.D[0][0]}: {e.D[0][1]}" for e in examples],
            index=0,
        )
        placeholder_lemma, placeholder_definition = (
            placeholder[: placeholder.index(":")],
            placeholder[placeholder.index(":") + 2 :],
        )

        # read input
        lemma = st.text_area("Input lemma:", placeholder_lemma)
        definition = st.text_area(
            "Input definition of the given lemma:", placeholder_definition
        )
        input_sample = ExmodSample(
            source_language="en", D=[(lemma, definition)], target_language="en"
        )

        # button
        should_disambiguate = st.button("Exemplify", key="classify")

        if should_disambiguate:

            # tag sentence
            time_start = time.perf_counter()
            predicted_sample = list(
                model.predict([input_sample], dataset_conf=dataset_conf)
            )[0]
            time_end = time.perf_counter()

            # extract example
            s, phi = predicted_sample.predicted_annotation[0]
            annotated_html_components = [
                str(html.escape(s[: phi[0][0]])),
                str(annotation(*(s[phi[0][0] : phi[0][-1] + 1], "", "#FFA726"))),
                str(html.escape(s[phi[0][-1] + 1 :])),
            ]

            # print
            st.markdown(
                "\n".join(
                    [
                        "<div>",
                        *annotated_html_components,
                        "<p></p>"
                        f'<div style="text-align: right"><p style="color: gray">Time: {(time_end - time_start):.2f}s</p></div>'
                        "</div>",
                    ]
                ),
                unsafe_allow_html=True,
            )

    demo()
    hiw()


if __name__ == "__main__":

    main(
        "experiments/bart-semcor-wne-fews/2023-02-25/14-30-47/checkpoints/best.ckpt",
        "configurations/prediction-params/beam.yaml",
        cuda_device=-1,
    )
