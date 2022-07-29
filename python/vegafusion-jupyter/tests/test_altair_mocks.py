from pathlib import Path
import jupytext
import io
import tempfile
from subprocess import Popen
import chromedriver_binary
import pytest
from selenium.webdriver import ActionChains
from selenium import webdriver
import time
from io import BytesIO
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
import json
import shutil
from tenacity import retry, wait, stop
import os
from flaky import flaky

here = Path(__file__).parent
altair_mocks_dir = here / "altair_mocks"
temp_notebooks_dir = here / "temp_notebooks"
temp_screenshots_dir = here / "temp_screenshot"

altair_markdown_template = r"""
```python
import altair as alt
alt.renderers.enable('default', embed_options={'actions': False});
```

```python
{code}
```

```python
assert(alt.renderers.active == "default")
assert(alt.data_transformers.active == 'default')
```
"""

vegafusion_feather_markdown_template = r"""
```python
import altair as alt
import vegafusion_jupyter as vf
vf.enable()
```

```python
{code}
```

```python
assert(alt.renderers.active == "vegafusion-jupyter")
assert(alt.data_transformers.active == 'vegafusion-feather')
```
"""

vegafusion_default_markdown_template = r"""
```python
import altair as alt
import vegafusion_jupyter as vf
vf.enable()
alt.data_transformers.enable("default");
```

```python
{code}
```

```python
assert(alt.renderers.active == "vegafusion-jupyter")
assert(alt.data_transformers.active == 'default')
```
"""


def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    # Initialize notebooks and screenshots to empty directories
    shutil.rmtree(temp_notebooks_dir, ignore_errors=True)
    temp_notebooks_dir.mkdir(parents=True, exist_ok=True)

    shutil.rmtree(temp_screenshots_dir, ignore_errors=True)
    temp_screenshots_dir.mkdir(parents=True, exist_ok=True)


@flaky(max_runs=2)
@pytest.mark.parametrize(
    "mock_name,img_tolerance,delay", [
        ("area/cumulative_count", 1.0, 0.5),
        ("area/density_facet", 1.0, 0.5),
        ("area/gradient", 1.0, 0.5),
        ("area/horizon_graph", 1.0, 0.5),
        ("area/layered", 1.0, 0.5),
        ("area/normalized_stacked", 1.0, 0.5),
        ("area/density_stack", 1.0, 0.5),
        ("area/trellis", 1.0, 0.5),
        ("area/trellis_sort_array", 1.0, 0.5),
        ("area/streamgraph", 0.998, 0.5),
        ("bar/with_highlighted_bar", 1.0, 0.5),
        ("bar/with_labels", 1.0, 0.5),
        ("bar/with_line_at_mean", 1.0, 0.5),
        ("bar/with_line_on_dual_axis", 1.0, 0.5),
        ("bar/with_rolling_mean", 1.0, 0.5),
        ("bar/with_rounded_edges", 0.999, 0.5),
        ("bar/and_tick_chart", 1.0, 0.5),
        ("bar/percentage_of_total", 1.0, 0.5),
        ("bar/trellis_compact", 1.0, 0.5),
        ("bar/diverging_stacked", 1.0, 0.5),
        ("bar/grouped", 1.0, 0.5),
        ("bar/horizontal", 1.0, 0.5),
        ("bar/horizontal_grouped", 1.0, 0.5),
        ("bar/horizontal_stacked", 0.999, 0.5),
        ("bar/normalized_stacked", 0.999, 0.5),
        ("bar/sorted", 1.0, 0.5),
        ("bar/stacked", 0.999, 0.5),
        ("bar/stacked_with_sorted_segments", 0.999, 0.5),
        ("bar/stacked_with_text_overlay", 0.999, 0.5),
        ("bar/trellis_stacked", 1.0, 0.5),
        ("bar/trellis_stacked", 1.0, 0.5),
        ("bar/with_negative_values", 1.0, 0.5),
        ("bar/layered", 1.0, 0.5),
        ("bar/with_error_bars", 0.998, 0.5),
        ("casestudy/co2_concentration", 1.0, 0.5),
        ("casestudy/gapminder_bubble_plot", 1.0, 0.5),
        ("casestudy/iowa_electricity", 1.0, 0.5),
        ("casestudy/natural_disasters", 1.0, 0.5),
        ("casestudy/top_k_with_others", 1.0, 0.5),
        ("casestudy/wheat_wages", 1.0, 0.5),
        ("casestudy/window_rank", 0.999, 0.5),
        ("casestudy/airports", 1.0, 0.5),
        ("casestudy/us_state_capitals", 1.0, 0.5),
        ("casestudy/falkensee", 1.0, 0.5),
        ("casestudy/us_employment", 1.0, 0.5),
        ("casestudy/top_k_items", 1.0, 0.5),

        # Different order of ticks for equal bar lengths
        ("casestudy/top_k_letters", 0.995, 0.5),
        ("casestudy/isotype", 1.0, 0.5),
        ("casestudy/london_tube", 1.0, 0.5),
        ("casestudy/isotype_emoji", 1.0, 0.5),
        ("casestudy/beckers_barley_trellis_plot", 1.0, 0.5),
        ("casestudy/anscombe_plot", 1.0, 0.5),
        ("casestudy/us_population_over_time_facet", 1.0, 0.5),
        ("casestudy/one_dot_per_zipcode", 0.999, 0.5),
        ("circular/donut", 1.0, 0.5),
        ("circular/pie", 1.0, 0.5),
        ("circular/pie_with_labels", 1.0, 0.5),
        ("circular/radial", 1.0, 0.5),
        ("circular/pacman", 1.0, 0.5),
        ("histogram/with_a_global_mean_overlay", 1.0, 0.5),
        ("histogram/layered", 1.0, 0.5),
        ("histogram/trellis", 1.0, 0.5),
        ("interactive/selection_layer_bar_month", 1.0, 1),
        ("interactive/area-interval_selection", 1.0, 1),
        ("interactive/layered_crossfilter", 1.0, 1),
        ("interactive/scatter_with_histogram", 1.0, 1),
        ("interactive/select_detail", 1.0, 1),
        ("interactive/scatter_plot", 1.0, 1),
        ("interactive/brush", 1.0, 1),
        ("interactive/multiline_tooltip", 1.0, 1),
        ("interactive/scatter_linked_brush", 1.0, 1),
        ("interactive/casestudy-us_population_pyramid_over_time", 1.0, 1),
        ("interactive/multiline_highlight", 1.0, 1),
        ("interactive/scatter-with_minimap", 1.0, 1),
        ("interactive/select_mark_area", 1.0, 1),
        ("interactive/legend", 0.998, 1),
        ("interactive/cross_highlight", 0.999, 1),
        ("interactive/selection_histogram", 1.0, 1),
        ("interactive/scatter-with_linked_table", 1.0, 1),
        ("interactive/scatter_with_layered_histogram", 1.0, 1),
        ("interactive/casestudy-seattle_weather_interactive", 1.0, 1),
        ("interactive/casestudy-us_population_over_time", 1.0, 1),
        ("interactive/scatter-href", 1.0, 1),
        ("interactive/other-image_tooltip", 1.0, 1),
        ("interactive/casestudy-weather_heatmap", 0.999, 2),
        ("interactive/casestudy-airport_connections", 1.0, 1),
        ("interactive/histogram-responsive", 1.0, 8),
        ("line/bump_chart", 0.999, 0.5),
        ("line/filled_step_chart", 1.0, 0.5),
        ("line/with_cumsum", 1.0, 0.5),
        ("line/with_logarithmic_scale", 1.0, 0.5),
        ("line/percent_axis", 1.0, 0.5),
        ("line/with_points", 1.0, 0.5),
        ("line/with_generator", 1.0, 0.5),
        ("line/slope_graph", 1.0, 0.5),
        ("line/slope_graph2", 0.999, 0.5),
        ("line/step_chart", 1.0, 0.5),
        ("line/layer_line_color_rule", 1.0, 0.5),
        ("line/multi_series", 1.0, 0.5),
        ("line/with_ci", 1.0, 0.5),
        ("line/trail_marker", 1.0, 0.5),
        ("line/with_datum", 1.0, 0.5),
        ("line/with_color_datum", 1.0, 0.5),
        ("maps/choropleth", 1.0, 0.5),
        ("maps/choropleth_repeat", 1.0, 0.5),
        ("maps/us_incomebrackets_by_state_facet", 1.0, 0.5),
        ("maps/world", 1.0, 0.5),
        ("maps/world_projections", 1.0, 0.5),
        ("maps/airports_count", 0.999, 0.5),
        ("other/bar_chart_with_highlighted_segment", 1.0, 0.5),
        ("other/beckers_barley_wrapped_facet", 1.0, 0.5),
        ("other/boxplot", 1.0, 0.5),
        ("other/comet_chart", 1.0, 0.5),
        ("other/errorbars_with_std", 1.0, 0.5),
        ("other/scatter_marginal_hist", 0.999, 0.5),
        ("other/gantt_chart", 1.0, 0.5),
        ("other/isotype_grid", 1.0, 0.5),
        ("other/layered_chart_with_dual_axis", 1.0, 0.5),
        ("other/ridgeline_plot", 1.0, 0.5),
        ("other/stem_and_leaf", 1.0, 0.5),
        ("other/layered_heatmap_text", 1.0, 0.5),
        ("other/candlestick_chart", 1.0, 0.5),
        ("other/multiple_marks", 1.0, 0.5),
        ("other/hexbins", 0.999, 0.5),
        ("other/wilkinson_dot_plot", 1.0, 0.5),
        ("other/binned_heatmap", 0.998, 0.5),
        ("other/normed_parallel_coordinates", 1.0, 0.5),
        ("other/parallel_coordinates", 1.0, 0.5),
        ("other/violin_plot", 1.0, 0.5),
        ("other/ranged_dot_plot", 1.0, 0.5),
        ("scatter/binned", 0.999, 0.5),
        ("scatter/bubble_plot", 1.0, 0.5),
        ("scatter/connected", 1.0, 0.5),
        ("scatter/dot_dash_plot", 1.0, 0.5),
        ("scatter/multifeature", 1.0, 0.5),
        ("scatter/poly_fit_regression", 1.0, 0.5),
        ("scatter/qq", 1.0, 0.5),
        ("scatter/matrix", 1.0, 0.5),
        ("scatter/with_lowess", 1.0, 0.5),
        ("scatter/with_errorbars", 1.0, 0.5),
        ("scatter/with_labels", 1.0, 0.5),
        ("scatter/table_bubble_plot_github", 0.999, 0.5),
        ("scatter/trellis", 1.0, 0.5),
        ("scatter/wind_vector_map", 1.0, 0.5),
        ("scatter/with_rolling_mean", 1.0, 0.5),
        ("simple/stacked_bar_chart", 1.0, 0.5),
        ("simple/bar_chart", 1.0, 0.5),
        ("simple/heatmap", 1.0, 0.5),
        ("simple/line_chart", 1.0, 0.5),
        ("simple/scatter_tooltips", 1.0, 0.5),
        ("simple/strip_chart", 1.0, 0.5),

        # Non-deterministic mocks have lower image tolerance
        ("other/errorbars_with_ci", 0.8, 0.5),
        ("other/sorted_error_bars_with_ci", 0.8, 0.5),
        ("scatter/stripplot", 0.8, 0.5)  # random()
    ])
def test_altair_mock(mock_name, img_tolerance, delay):

    # Build Jupytext Markdown text containing the mock's code
    mock_path = altair_mocks_dir / mock_name / "mock.py"
    actions = load_actions(mock_name)

    mock_code = mock_path.read_text()
    altair_markdown = altair_markdown_template.replace("{code}", mock_code)
    vegafusion_arrow_markdown = vegafusion_feather_markdown_template.replace("{code}", mock_code)
    vegafusion_default_markdown = vegafusion_default_markdown_template.replace("{code}", mock_code)

    # Use jupytext to convert markdown to an ipynb file
    altair_notebook = jupytext.read(io.StringIO(altair_markdown), fmt="markdown")
    vegafusion_arrow_notebook = jupytext.read(io.StringIO(vegafusion_arrow_markdown), fmt="markdown")
    vegafusion_default_notebook = jupytext.read(io.StringIO(vegafusion_default_markdown), fmt="markdown")

    # Create selenium Chrome instance
    chrome_opts = webdriver.ChromeOptions()
    if os.environ.get("VEGAFUSION_TEST_HEADLESS"):
        chrome_opts.add_argument("--headless")
    chrome_driver = webdriver.Chrome(options=chrome_opts)
    chrome_driver.set_window_size(800, 800)

    # Launch Voila server
    voila_proc = Popen(["voila", "--no-browser", "--enable_nbextensions=True"], cwd=temp_notebooks_dir)

    # Sleep to allow Voila itself to start (this does not include loading a particular dashboard).
    time.sleep(1.5)

    try:
        name = mock_name.replace("/", "-")
        altair_imgs = export_image_sequence(
            chrome_driver, altair_notebook, name + "_altair", actions, delay
        )
        vegafusion_arrow_imgs = export_image_sequence(
            chrome_driver, vegafusion_arrow_notebook, name + "_vegafusion_feather", actions, delay
        )
        vegafusion_default_imgs = export_image_sequence(
            chrome_driver, vegafusion_default_notebook, name + "_vegafusion", actions, delay
        )

        for i in range(len(altair_imgs)):
            altair_img = altair_imgs[i]
            vegafusion_arrow_img = vegafusion_arrow_imgs[i]
            vegafusion_default_img = vegafusion_default_imgs[i]

            assert altair_img.shape == vegafusion_arrow_img.shape, "Size mismatch with Arrow data transformer"
            assert altair_img.shape == vegafusion_default_img.shape, "Size mismatch with default data transformer"

            similarity_arrow_value = ssim(altair_img, vegafusion_arrow_img, channel_axis=2)
            similarity_default_value = ssim(altair_img, vegafusion_default_img, channel_axis=2)

            print(f"({i}) similarity_arrow_value={similarity_arrow_value}")
            print(f"({i}) similarity_default_value={similarity_default_value}")

            assert similarity_arrow_value >= img_tolerance, f"Similarity failed with Arrow data transformer on image {i}"
            assert similarity_default_value >= img_tolerance, f"Similarity failed with default data transformer on image {i}"

    finally:
        voila_proc.kill()
        chrome_driver.close()
        time.sleep(0.25)


def load_actions(mock_name):
    actions_path = altair_mocks_dir / mock_name / "actions.json"
    if actions_path.exists():
        return json.loads(actions_path.read_text())
    else:
        return [{"type": "snapshot"}]


def export_image_sequence(
        chrome_driver: webdriver.Chrome,
        notebook: jupytext.jupytext.NotebookNode,
        name,
        actions,
        delay,
        voila_url_base: str = "http://localhost:8866/voila/render/",
):
    imgs = []

    with tempfile.NamedTemporaryFile(
            mode="wt", dir=temp_notebooks_dir, suffix=".ipynb",
    ) as f:
        jupytext.write(notebook, f, fmt="ipynb")
        f.file.flush()

        temp_file_path = Path(f.name)

        # Server file with voila
        url = voila_url_base.rstrip("/") + "/" + temp_file_path.name

        # Open url with selenium
        # Get canvas element (the canvas that Vega renders to)
        @retry(wait=wait.wait_fixed(0.5), stop=stop.stop_after_delay(10))
        def get_url():
            return chrome_driver.get(url)
        get_url()

        # Remove padding, margins, and standardize line height.
        css = ("body, .jp-Cell, .jp-Notebook, .jupyter-widgets, .jp-RenderedHTMLCommon "
               "{margin: 0 !important; padding: 0 !important; line-height: 1.3 !important;}")
        script = 'document.styleSheets[0].insertRule("' + css + '", 0 )'
        chrome_driver.execute_script(script)

        # Hide vegafusion logo for comparison with Altair
        css = ".vegafusion-embed summary { visibility: hidden }"
        script = 'document.styleSheets[0].insertRule("' + css + '", 0 )'
        chrome_driver.execute_script(script)

        # Get canvas element (the canvas that Vega renders to)
        @retry(wait=wait.wait_fixed(0.5), stop=stop.stop_after_delay(10))
        def get_canvas():
            return chrome_driver.find_element_by_xpath("//canvas")

        canvas = get_canvas()
        time.sleep(delay)

        # Process actions
        chain = ActionChains(chrome_driver)
        for i, action in enumerate(actions):
            action_type = action["type"]
            if action_type in ("snapshot", "screenshot"):
                time.sleep(0.5)
                chain.perform()
                time.sleep(1.0)

                img_path = (temp_screenshots_dir / f"{name}_{i}.png").as_posix();
                print(f"img_path: {img_path}")
                if action_type == "snapshot":
                    img_bytes = canvas.screenshot_as_png
                    # Write to file for debugging
                    canvas.screenshot(img_path)
                else:
                    img_bytes = chrome_driver.get_screenshot_as_png()
                    chrome_driver.save_screenshot(img_path)

                # Get png representation in binary (bytes) from driver
                # convert this into a 3D numpy image
                img = imread(BytesIO(img_bytes))
                imgs.append(img)

                # Reset chain
                chain = ActionChains(chrome_driver)

            elif action_type == "move_to":
                coords = action["coords"]
                chain = chain.move_to_element_with_offset(canvas, coords[0], coords[1])
            elif action_type == "move_by":
                coords = action["coords"]
                chain = chain.move_by_offset(coords[0], coords[1])
            elif action_type == "click_and_hold":
                chain = chain.click_and_hold()
            elif action_type == "click":
                chain = chain.click()
            elif action_type == "double_click":
                chain = chain.double_click()
            elif action_type == "release":
                chain = chain.release()
            else:
                raise ValueError(f"Invalid action type {action_type}")

    return imgs
