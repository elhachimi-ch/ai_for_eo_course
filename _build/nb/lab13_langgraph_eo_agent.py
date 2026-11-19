# %% [markdown]
# # Lab 10 — A Sentinel-2 agent with LangGraph
# **Deep Learning for Earth Observation** · Part IX
#
# An **agent** is a loop: the LLM *reasons* and decides to call a tool, *your code acts*, the result is *observed*, and the loop repeats until the model answers.
# LangGraph makes this loop an explicit **state graph**: `START → agent ⇄ tools → END`, with memory (checkpointer) and optional human approval (interrupt).
#
# Tools we give the agent:
# 1. `geocode(place)` → bounding box (OpenStreetMap Nominatim; respect its usage policy — 1 request/s, identify your app).
# 2. `search_sentinel2(bbox, start, end, max_cloud)` → list of scenes (Microsoft Planetary Computer STAC).
# 3. `mean_ndvi(item_id, bbox)` → cloud-masked mean NDVI of one scene.
#
# Requires a **tool-calling** chat model: a hosted model (API key) or a local Ollama model that supports tools (e.g. `qwen2.5:7b`).

# %%
!pip -q install langgraph langchain langchain-core pystac-client planetary-computer odc-stac geopy
# local model option:  !pip -q install langchain-ollama

# %%
import numpy as np
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
import pystac_client, planetary_computer, odc.stac
from geopy.geocoders import Nominatim

MODEL = "openai:gpt-4o-mini"            # or "ollama:qwen2.5:7b", "anthropic:<model-name>", ...
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
cat = pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace)
geolocator = Nominatim(user_agent="dl4eo-course-lab10")

# %% [markdown]
# ## 1 · Tools — the docstring and type hints ARE the API the model sees

# %%
@tool
def geocode(place: str) -> list[float]:
    """Return the bounding box [W, S, E, N] in degrees of a named place (city, region, plain)."""
    loc = geolocator.geocode(place, exactly_one=True)
    if loc is None:
        raise ValueError(f"place not found: {place}")
    s, n, w, e = map(float, loc.raw["boundingbox"])
    # keep requests light: shrink very large boxes around the centre (max ~0.3 degrees)
    cx, cy = (w + e) / 2, (s + n) / 2
    hw, hh = min((e - w) / 2, 0.15), min((n - s) / 2, 0.15)
    return [round(cx - hw, 4), round(cy - hh, 4), round(cx + hw, 4), round(cy + hh, 4)]

@tool
def search_sentinel2(bbox: list[float], start: str, end: str, max_cloud: int = 20) -> str:
    """List up to 10 Sentinel-2 L2A scenes over bbox [W,S,E,N] acquired between start and end
    (YYYY-MM-DD) with cloud cover below max_cloud percent, clearest first."""
    items = cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{start}/{end}",
                       query={"eo:cloud_cover": {"lt": max_cloud}}).item_collection()
    items = sorted(items, key=lambda i: i.properties["eo:cloud_cover"])[:10]
    if not items:
        return "no scenes found"
    return "\n".join(f"{i.id} date={i.datetime:%Y-%m-%d} cloud={i.properties['eo:cloud_cover']:.1f}%" for i in items)

@tool
def mean_ndvi(item_id: str, bbox: list[float]) -> float:
    """Mean NDVI of one Sentinel-2 L2A scene (by item id) inside bbox [W,S,E,N], clouds and shadows masked."""
    item = cat.get_collection("sentinel-2-l2a").get_item(item_id)
    epsg = item.properties.get("proj:epsg") or int(str(item.properties.get("proj:code", "EPSG:32629")).split(":")[-1])
    ds = odc.stac.load([item], bands=["B04", "B08", "SCL"], bbox=bbox, resolution=20, crs=f"EPSG:{epsg}").isel(time=0)
    baseline = float(item.properties.get("s2:processing_baseline", "0") or 0)
    off = 1000 if baseline >= 4.0 else 0                       # radiometric offset since baseline 04.00
    red = (ds.B04.astype("float32") - off) / 1e4
    nir = (ds.B08.astype("float32") - off) / 1e4
    ok = ~ds.SCL.isin([0, 1, 3, 8, 9, 10, 11])                 # no-data, saturated, shadow, clouds, cirrus, snow
    ndvi = ((nir - red) / (nir + red)).where(ok)
    return round(float(ndvi.mean(skipna=True)), 4)

tools = [geocode, search_sentinel2, mean_ndvi]
print(mean_ndvi.args)          # what the model will see

# %% [markdown]
# ## 2 · The graph

# %%
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

SYSTEM = ("You are an Earth-observation assistant. Use the tools to get data; never invent scene ids or numbers. "
          "Pick the clearest scene per period. Report scene ids, dates, cloud cover and caveats.")
llm = init_chat_model(MODEL, temperature=0).bind_tools(tools)

def agent(state: MessagesState):
    return {"messages": [llm.invoke([("system", SYSTEM)] + state["messages"])]}

g = StateGraph(MessagesState)
g.add_node("agent", agent)
g.add_node("tools", ToolNode(tools))
g.add_edge(START, "agent")
g.add_conditional_edges("agent", tools_condition)      # -> "tools" if tool calls, else END
g.add_edge("tools", "agent")
app = g.compile(checkpointer=InMemorySaver())

try:
    from IPython.display import Image, display
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception as e:
    print(app.get_graph().draw_ascii() if hasattr(app.get_graph(), "draw_ascii") else e)

# %% [markdown]
# ## 3 · Run — and read the reason → act → observe trace

# %%
cfg = {"configurable": {"thread_id": "haouz"}}
q = ("Compare mean NDVI over the Haouz plain near Marrakech in mid-April 2024 versus mid-April 2025, "
     "using the clearest scene in each period. Is vegetation greener in 2025?")
out = app.invoke({"messages": [("user", q)]}, cfg)
for m in out["messages"]:
    kind = type(m).__name__
    if getattr(m, "tool_calls", None):
        print(f"[{kind}] TOOL CALLS:", [(c["name"], c["args"]) for c in m.tool_calls])
    else:
        print(f"[{kind}]", str(m.content)[:600])
    print("-" * 80)

# %%
# Memory: the same thread remembers the previous exchange
out = app.invoke({"messages": [("user", "Now do the same for the Saïss plain near Meknès in April 2025 only.")]}, cfg)
print(out["messages"][-1].content)

# %% [markdown]
# ## 4 · Human-in-the-loop (optional)
# Compile with `interrupt_before=["tools"]` to pause before every tool call; inspect the planned call, then resume.

# %%
app_hitl = g.compile(checkpointer=InMemorySaver(), interrupt_before=["tools"])
cfg2 = {"configurable": {"thread_id": "review"}}
state = app_hitl.invoke({"messages": [("user", "Find clear Sentinel-2 scenes over Agadir in March 2025.")]}, cfg2)
print("Planned tool calls:", state["messages"][-1].tool_calls)
# approve and continue:
state = app_hitl.invoke(None, cfg2)
print(state["messages"][-1].content if not getattr(state["messages"][-1], "tool_calls", None) else "paused again before tools")

# %% [markdown]
# ## Exercises
# 1. Add a tool `ndvi_timeseries(bbox, year)` that returns monthly mean NDVI. How does the agent decide when to use it?
# 2. Give the agent a wrong instruction (“use scene id XYZ”) and observe how tool errors are handled.
# 3. Add a `report` node that writes the final answer in a fixed template (place, dates, scenes, values, caveats).
# 4. Which steps of your own EO workflow could an agent automate — and which should always stay with a human?
