# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Project hooks."""
from typing import Any, Dict, Iterable, Optional

from kedro.config import TemplatedConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.versioning import Journal
from kedro_mlflow.pipeline import pipeline_ml_factory

from kedro_mlflow_tutorial import __version__ as PROJECT_VERSION
from kedro_mlflow_tutorial.pipelines.etl_app.pipeline import create_etl_pipeline
from kedro_mlflow_tutorial.pipelines.ml_app.pipeline import create_ml_pipeline
from kedro_mlflow_tutorial.pipelines.user_app.pipeline import create_user_app_pipeline


class ProjectHooks:
    @hook_impl
    def register_pipelines(self) -> Dict[str, Pipeline]:
        """Register the project's pipeline.

        Returns:
            A mapping from a pipeline name to a ``Pipeline`` object.

        """
        etl_pipeline = create_etl_pipeline()
        etl_instances_pipeline = etl_pipeline.only_nodes_with_tags("etl_instances")
        etl_labels_pipeline = etl_pipeline.only_nodes_with_tags("etl_labels")

        ml_pipeline = create_ml_pipeline()
        inference_pipeline = ml_pipeline.only_nodes_with_tags("inference")
        training_pipeline_ml = pipeline_ml_factory(
            training=ml_pipeline.only_nodes_with_tags("training"),
            inference=inference_pipeline,
            input_name="instances",
            model_name="kedro_mlflow_tutorial",
            conda_env={
                "python": 3.7,
                "pip": [f"kedro_mlflow_tutorial=={PROJECT_VERSION}"],
            },
        )

        user_app_pipeline = create_user_app_pipeline()

        return {
            "etl_instances": etl_instances_pipeline,
            "etl_labels": etl_labels_pipeline,
            "training": training_pipeline_ml,
            "inference": inference_pipeline,
            "user_app": user_app_pipeline,
            "__default__": etl_instances_pipeline
            + etl_labels_pipeline
            + inference_pipeline
            + user_app_pipeline,
        }

    @hook_impl
    def register_config_loader(
        self, conf_paths: Iterable[str]
    ) -> TemplatedConfigLoader:
        return TemplatedConfigLoader(
            conf_paths,
            globals_pattern="*globals.yml",
            globals_dict={},
        )

    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
        journal: Journal,
    ) -> DataCatalog:
        return DataCatalog.from_config(
            catalog, credentials, load_versions, save_version, journal
        )


project_hooks = ProjectHooks()
