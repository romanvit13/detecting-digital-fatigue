from digital_fatigue.ui import create_app


def main():
    app = create_app()
    app.launch(share=True, debug=True)


if __name__ == "__main__":
    main()
