import { SwishjamProvider } from "@swishjam/react";

export default function SwishJam({ children }) {
	console.log("Swishjam loaded");

	return <SwishjamProvider apiKey="swishjam_prdct-a204a99a8777f463">{children}</SwishjamProvider>;
}
