import httpx
from langchain_core.tools import tool

from ..config import get_settings

settings = get_settings()


@tool
def get_route_options(departure: str, arrival: str) -> str:
    """
    Search route options (train / bus / airplane) between two locations
    using the Spring Boot backend.
    """
    url = f"{settings.spring_base_url}/api/routes/options"
    print(f"[TOOL] get_route_options called with departure={departure!r}, arrival={arrival!r}")
    print(f"[TOOL] HTTP GET {url}")

    try:
        resp = httpx.get(
            url,
            params={"departure": departure, "arrival": arrival},
            timeout=10.0,
        )
    except Exception as e:
        msg = f"[EROARE_TOOL] Nu am putut face request la {url}: {type(e).__name__}: {e!r}"
        print(msg)
        return msg

    print(f"[TOOL] Server response status: {resp.status_code}")
    print(f"[TOOL] Server raw body: {resp.text!r}")

    if resp.status_code != 200:
        msg = (
            f"[EROARE_TOOL] Serverul Spring a raspuns cu status {resp.status_code} "
            f"pentru {departure} -> {arrival}. Body: {resp.text}"
        )
        print(msg)
        return msg

    try:
        data = resp.json()
    except Exception as e:
        msg = (
            f"[EROARE_TOOL] Raspunsul de la server nu este JSON valid: {type(e).__name__}: {e!r}. "
            f"Body brut: {resp.text}"
        )
        print(msg)
        return msg

    if not data:
        msg = f"Nu sunt rute disponibile pentru {departure} -> {arrival}"
        print(f"[TOOL] {msg}")
        return msg

    lines = [f"Am gasit {len(data)} optiuni de rute pentru {departure} -> {arrival}:\n"]

    for i, option in enumerate(data[:5], start=1):
        legs = option.get("legs", [])
        if not legs:
            continue

        lines.append(f"Optiunea {i}:")
        for j, leg in enumerate(legs, start=1):
            r_type = leg.get("transportType", "UNKNOWN")
            r_id = leg.get("routeId", "?")
            t_id = leg.get("transportId", "?")
            stops = leg.get("stops", [])
            seats = leg.get("availableSeats", "?")

            route_path = " -> ".join(stops) if stops else "N/A"
            lines.append(
                f"  Segment {j}: [{r_type}] {t_id} (routeId={r_id})\n"
                f"    Traseu: {route_path}\n"
                f"    Locuri disponibile: {seats}"
            )

        lines.append("")

    result = "\n".join(lines)
    print(f"[TOOL] get_route_options result:\n{result}")
    return result
