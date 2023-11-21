/* Copyright (C) 2019-2023 Battelle Memorial Institute
 * loadshedCommNetwork.cc
 *
 *  Created on: Sep 25, 2019
 *      Author: afisher
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/ipv4-global-routing-helper.h"

#include "ns3/helics-helper.h"

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <string>


using namespace ns3;

NS_LOG_COMPONENT_DEFINE("loadshedCommNetwork");

int main(int argc, char *argv[])
{
	//Handle input arguments
	std::string helicsConfigFile = "";
	double simulationRunTime = 0.0;
	CommandLine cmd;
	cmd.AddValue("helicsConfigFile", "The helics configuration file to use.", helicsConfigFile);
	cmd.AddValue("simulationRunTime", "The simulation run time.", simulationRunTime);
	cmd.Parse(argc, argv);
	if(helicsConfigFile.compare("") == 0){
		std::cerr << "No helics configuration file was given." << std::endl;
		return 1;
	}
	if(simulationRunTime <= 0.0){
		std::cerr << "No simulation run time was given." << std::endl;
		return 1;
	}
	else {
		std::cout << "Simulation time " << simulationRunTime << " seconds." << std::endl;
	}
	//Create HELICS federate
	HelicsHelper helicsHelper;
	helicsHelper.SetupApplicationFederateWithConfig(helicsConfigFile);

	//Figure out How many P2P nodes we need
	int nNodes = helics_federate->getEndpointCount();
	NodeContainer p2pNodes;
	p2pNodes.Create(nNodes);
	PointToPointHelper pointToPoint;
	pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
	pointToPoint.SetChannelAttribute ("Delay", StringValue ("20s")); //200 ms
	NetDeviceContainer p2pDevices;
	p2pDevices = pointToPoint.Install (p2pNodes);
	InternetStackHelper stack;
	for(int i=0; i < nNodes; i++) {
		stack.Install(p2pNodes.Get(i));
	}
	Ipv4AddressHelper address;
	address.SetBase ("10.1.1.0", "255.255.255.0");
	Ipv4InterfaceContainer p2pInterfaces;
	p2pInterfaces = address.Assign (p2pDevices);

	// Attach Helics Application to nodes.
	std::cout << "Running " << nNodes << " nodes to end time " << Seconds(simulationRunTime) << std::endl;
	std::vector<ApplicationContainer> helicsFilterApps;
	for(int i=0; i<nNodes; i++) {
		std::cout << ":Filter:" << helics_federate->getFilter(i).getName() << ":Endpoint:" << helics_federate->getEndpoint(i).getName() << std::endl;
		ApplicationContainer apps = helicsHelper.InstallFilter(p2pNodes.Get(i), helics_federate->getFilter(i), helics_federate->getEndpoint(i));
		apps.Start(Seconds(0.0));
		apps.Stop(Seconds(simulationRunTime));
		helicsFilterApps.push_back(apps);
	}
	Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

	Simulator::Stop(Seconds(simulationRunTime));
	std::cout << "===== Run simulator .... =====" << std::endl;
	Simulator::Run();
    std::cout << "===== Destroy simulator. =====" << std::endl;
	Simulator::Destroy();
	return 0;
}