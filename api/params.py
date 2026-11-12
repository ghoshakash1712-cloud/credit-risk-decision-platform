from api.errors import ApiError

def parse_params(args: dict, rules: dict) -> dict:
    # AP-04: typos and unknown parameters are rejected with 400 invalid_parameter
    unknown = set(args.keys()) - set(rules.keys())
    if unknown:
        first_un = sorted(list(unknown))[0]
        raise ApiError(
            code="invalid_parameter",
            status_code=400,
            message=f"Unknown parameter: {first_un}",
            details={"parameter": first_un}
        )

    out = {}
    for name, rule in rules.items():
        raw = args.get(name)
        if raw is None or raw == "":
            if rule.get("required", False):
                raise ApiError(
                    code="invalid_parameter",
                    status_code=400,
                    message=f"{name} is required",
                    details={"parameter": name}
                )
            out[name] = rule.get("default")
            continue

        p_type = rule.get("type", str)
        try:
            if p_type == float:
                val = float(raw)
            elif p_type == int:
                val = int(raw)
            else:
                val = str(raw).strip()
        except ValueError:
            raise ApiError(
                code="invalid_parameter",
                status_code=400,
                message=f"{name} must be of type {p_type.__name__}",
                details={"parameter": name}
            )

        # Range checks
        if "min" in rule and val < rule["min"]:
            raise ApiError(
                code="invalid_parameter",
                status_code=400,
                message=f"{name} must be between {rule['min']} and {rule['max']}",
                details={"parameter": name}
            )
        if "max" in rule and val > rule["max"]:
            raise ApiError(
                code="invalid_parameter",
                status_code=400,
                message=f"{name} must be between {rule['min']} and {rule['max']}",
                details={"parameter": name}
            )

        # Allowed choices
        if "choices" in rule and val not in rule["choices"]:
            raise ApiError(
                code="invalid_parameter",
                status_code=400,
                message=f"{name} must be one of {rule['choices']}",
                details={"parameter": name}
            )

        out[name] = val

    return out
