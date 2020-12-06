def get_path(args):
    character_encoder = (
        f"{args.character_encoder}(beta={args.beta})"
        if args.character_encoder == "BetaVAE"
        else f"{args.character_encoder}"
    )
    hparam = f"encode_dim{args.encode_dim}"

    if args.da == "wt":
        da = f"da=wt({args.wildcard_ratio})"
    elif args.da == "ssa":
        da = f"da=ssa({args.gamma})"
    else:
        da = "da=none"

    character_encoder_name = f"character_encoder/{character_encoder}/{hparam}"

    classification_name = f"classification/{args.dataset}/{args.classification}/{character_encoder}/{hparam}/{da}"

    return {
        "save_dir": "../logs",
        "font": "../data/font/ipag.ttf",
        "char2embedding": f"../logs/{character_encoder_name}/version_{args.character_encoder_version}/CAE.pkl",
        "ja_chars": "../data/pickle/ja_chars.pkl",
        "newspaper": {
            "train": "../data/pickle/train_newspaper.pkl",
            "test": "../data/pickle/test_newspaper.pkl",
        },
        "livedoor": {
            "train": "../data/pickle/train_livedoor.pkl",
            "test": "../data/pickle/test_livedoor.pkl",
        },
        "character_encoder": {"name": character_encoder_name},
        "classification": {"name": classification_name},
    }
